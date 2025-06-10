import subprocess
import sys
import platform
import psutil
import socket
import os
from datetime import datetime
import requests
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from tavily import TavilyClient
from src import ASCIIArts, LLMModels, SystemEmojis, Prompts
from dotenv import load_dotenv


load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class SystemInfo:
    """Collect system information for the LLM context"""

    @staticmethod
    def get_system_info() -> dict:
        info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "os": platform.system(),
            "os_version": platform.version(),
            "distro flavour": "Kubuntu",
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "python_version": sys.version,
            "cpu_cores": psutil.cpu_count(),
            "total_memory": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
            "available_memory": f"{psutil.virtual_memory().available / (1024 ** 3):.2f} GB",
            "used_memory": f"{(psutil.virtual_memory().total / (1024 ** 3) - psutil.virtual_memory().available / (1024 ** 3)):.2f} GB",
            "disk_usage": f"{psutil.disk_usage('/').percent}%"
        }

        return info

    @staticmethod
    def get_temperatures() -> str:
        response = ""
        temps = psutil.sensors_temperatures()
        response +=  "📊 **System temperatures:**\n"

        if temps:
            for sensor_name, readings in temps.items():
                response += f"**{sensor_name}:**\n"
                for reading in readings:
                    response += f"  - {reading.label}: {reading.current}°C"
                    if reading.high:
                        response += f" (Máx: {reading.high}°C)"
                    if reading.critical:
                        response += f" (Crítico: {reading.critical}°C)"
                    response += "\n"
            response += "\n"
        else:
            response += "⚠️ Couldn't get system temperatures.\n\n"
        return response

class LLMClient:
    """Client for communicating with local LLM"""

    def __init__(self, base_url="http://localhost:11434", model_name="qwen3:8b", offline="False"):
        self.base_url = base_url
        self.model_name = model_name
        self.offline = offline

    @tool
    def get_hardware_temperatures() -> str:
        """Get current hardware temperatures (CPU, disk, etc.).
        ONLY use this function when the user specifically asks about hardware temperatures or thermal status."""

        return SystemInfo.get_temperatures()
    
    @tool
    def find_file(filename:str, path:str, options=None) -> str:
        """Find file in specified path
        INPUTS:
            - filename (str): name of the file to search
            - path (str): path to search in
            - options (list): optional arguments for the find command
        OUTPUT:
            - A string with a list of found coincidences.
        """

        result = ""
        cmd = ['find', path, '-name', filename]

        # Add optional arguments to the search
        if options:
            cmd.extend(options)

        try:
            # Ejecutar el comando
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
                shell=True
            )

            files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            return str([f for f in files if f])

        except subprocess.TimeoutExpired:
            return "Timeout Error"
        except subprocess.CalledProcessError as e:
            return f"Error running find command: {e.stderr}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
        
    @tool
    def search_query(query: str) -> str:
        """Search for a query in Internet"""
        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=True
        )
        return response["answer"]

    def check_ollama_status(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_response_stream(self, system_context:str, message_history:list):
        """Send prompt to LLM and get response as generator"""
        if not self.check_ollama_status():
            yield  "❌ Error: Ollama server is not running. Please start Ollama first."
            return

        message_history.insert(0, {
            'role': 'system',
            'content': system_context
        })

        message_history.insert(-1, {
            "role": "system", 
            "content": "Ignore previous questions. Use them only as context for understanding your environment, "
                        "but DONT'T answer them again. Answer ONLY the next question:"
        })

        tools = [self.get_hardware_temperatures, self.find_file] if self.offline else [self.get_hardware_temperatures, self.find_file, self.search_query]

        try:
            llm = ChatOllama(
                model=self.model_name,
                stream=True,
                temperature=0
            ).bind_tools(tools)

            response = llm.stream(message_history)

            chunk_count = 0
            tool_response = ""
            for chunk in response:
                chunk_count += 1
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        yield f"\n🔧 **Running tool:** {tool_call['name']}\n"

                        # Ejecutar la función y capturar el resultado
                        if tool_call['name'] == 'get_hardware_temperatures':
                            try:
                                tool_response = self.get_hardware_temperatures.invoke(None)

                            except Exception as e:
                                yield f"❌ Error getting temperatures: {str(e)}\n\n"
                        if tool_call['name'] == 'find_file':
                            try:
                                tool_response = self.find_file.invoke(tool_call['args'])
                            except Exception as e:
                                yield f"❌ Error finding files: {str(e)}\n\n"
                        if tool_call['name'] == "search_query":
                            try:
                                tool_response = self.search_query.invoke(tool_call['args'])
                            except Exception as e:
                                yield f"❌ Error getting answer from web: {str(e)}\n\n"
                if (self.model_name == 'qwen3:14b' or self.model_name == 'qwen3:8b') and chunk_count > 4 and chunk.content not in ['<think>', '</think>']:
                    yield chunk.content
                elif self.model_name != 'qwen3:14b' and self.model_name != "qwen3:8b":
                    yield chunk.content

            if tool_response:
                message_history.append(tool_response)
                response = llm.stream(message_history)
                chunk_count = 0
                for chunk in response:
                    chunk_count += 1
                    if (self.model_name == 'qwen3:14b' or self.model_name == 'qwen3:8b') and chunk_count > 4 and chunk.content not in ['<think>', '</think>']:
                        yield chunk.content
                    elif self.model_name != 'qwen3:14b' and self.model_name != "qwen3:8b":
                        yield chunk.content

        except Exception as e:
            yield f"❌ Error communicating with LLM: {str(e)}"
            return

class TerminalAssistant:
    """Main terminal assistant class"""

    def __init__(self, message_history=[]):

        self.system_info = SystemInfo()
        self.model_name = self._select_model()
        self.message_history = message_history
        art, emoji = self._select_art()
        self.art = art
        self.system_emoji = emoji
        self.offline = self._select_offline()
        self.system_context = self._build_system_context()
        self.llm_client = LLMClient(model_name=self.model_name, offline=self.offline)

    def _build_system_context(self):
        """Build system context for LLM"""
        info = self.system_info.get_system_info()
        context = Prompts.context_prompt_offline if self.offline else Prompts.context_prompt_online

        context += "\n SYSTEM INFORMATION: "

        for item in info.keys():
            context += f"\n- {item}: {info[item]}"
        
        return context
    
    def _select_model(self):
        LLMModels.print_available_models()
        model_name = input("\033[92m> Enter model: \033[0m")

        while model_name not in LLMModels.available_models:
            print("\033[93m! Model not available, select again")
            model_name = input("\033[92m> Enter model: \033[0m")

        return LLMModels.available_models[model_name]

    def _select_art(self):
        print("\n")
        SystemEmojis.print_available_emojis()
        art_style = input("\033[92m> Enter art style: \033[0m")

        while art_style not in SystemEmojis.available_emojis:
            print("\033[93m! Art style not available, select again")
            art_style = input("\033[92m> Enter model name: \033[0m")

        return [ASCIIArts.available_arts[art_style], SystemEmojis.available_emojis[art_style]]

    def _select_offline(self):
        print("\n")
        offline = input("\033[92m> Offline mode? (\033[93my/n\033[92m): \033[0m")

        while offline not in ["yes", "y", "no", "n"]:
            offline = input("\033[92m> Offline mode? (\033[93my/n\033[93m): \033[0m")

        if offline.lower() in ["yes", "y"]:
            return True
        elif offline.lower() in ["no", "n"]:
            return False

    def display_welcome(self):
        """Display welcome message with ASCII art"""
        print("\033[2J\033[H")  # Clear screen
        print("\033[92m" + self.art + "\033[0m")  # Green color
        print(f"\033[94mSystem: {platform.system()} {platform.release()}\033[0m")
        print(f"\033[94mHostname: {socket.gethostname()}\033[0m")
        print(f"\033[94mTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
        print(f"\033[94mModel: \033[95m{self.model_name}\033[0m")
        print("\033[93m" + "=" * 50 + "\033[0m")
        print("'help': commands info | 'exit': quit")
        print("\033[93m" + "=" * 50 + "\033[0m\n")

    def run_interactive_session(self):
        """Run the interactive chat session"""
        self.display_welcome()

        while True:
            try:
                user_input = input("\033[96m> \033[0m").strip()

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\033[92mGoodbye! 👋\033[0m")
                    sys.exit(0)
                elif user_input.lower() == 'help':
                    self.show_help()
                elif user_input.lower() == 'sysinfo':
                    self.show_system_info()
                elif user_input.lower() == 'clear':
                    print("\033[2J\033[H")
                    self.display_welcome()
                elif user_input.lower() == 'chmodel':
                    self.model_name = self._select_model()
                    self.llm_client = LLMClient(model_name=self.model_name)
                    self.display_welcome()
                elif user_input:
                    prompt = {'role': 'user', 'content': f"{user_input}"}
                    self.message_history.append(prompt)
                    print(f"\033[92m{self.system_emoji}: ", end="", flush=True)
                    for response in self.llm_client.get_response_stream(self.system_context, self.message_history):
                        print(response, end="", flush=True)
                    print("\033[0m")

            except KeyboardInterrupt:
                print("\n\033[92mGoodbye! 👋\033[0m")
                break
            except EOFError:
                break

    def show_help(self):
        """Show available commands"""
        help_text = """
        \033[94mAvailable Commands:\033[0m
        - help: Show this help message
        - sysinfo: Display detailed system information
        - clear: Clear the screen
        - exit/quit/bye: Exit the assistant
        - chmodel: Change the LLM model
        
        \033[94mYou can also ask me questions about:\033[0m
        - System administration
        - Programming help
        - File operations
        - Network troubleshooting
        - General assistance
        """

        print(help_text)

    def show_system_info(self):
        """Display detailed system information"""
        
        info = self.system_info.get_system_info()
        print("\033[94mSystem Information:\033[0m")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()

def launch_konsole():
    """Launch konsole terminal instance and run the assistant"""

    script_path = os.path.abspath(__file__)

    try:
        subprocess.run(['konsole', '-e', 'python3', script_path, '--interactive'])
    except FileNotFoundError:
        print("❌ Error: Konsole not found. Please install it first:")
        print("sudo apt install konsole")
        sys.exit(1)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        assistant = TerminalAssistant()
        assistant.run_interactive_session()
    else:
        launch_konsole()

if __name__ == "__main__":
    main()