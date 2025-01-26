import os
import asyncio
import subprocess
from sys import argv
from pathlib import Path
from pprint import pprint
import requests
from result_companion.parsers.cli_parser import parse_args
from result_companion.parsers.config import ConfigLoader
from result_companion.parsers.result_parser import get_robot_results_from_file_as_dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


def check_ollama_installed(ollama_version: list = ["ollama", "--version"]) -> None:
    try:
        result = subprocess.run(ollama_version, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Ollama is not installed.")
        print(f"Ollama version: {result.stdout.strip()}")
    except FileNotFoundError:
        raise Exception("Ollama is not installed.")
    

def check_http_server_running(url="http://localhost:8000") -> bool:
    # TODO: fix this check
    return True
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("HTTP server is running.")
        else:
            raise Exception("HTTP server is not responding correctly.")
    except requests.ConnectionError:
        raise Exception("HTTP server is not running.")


def start_ollama_server(start_cmd: list=["ollama", "start"]) -> None:
    try:
        subprocess.Popen(start_cmd)
        print("Starting Ollama server...")
    except Exception as e:
        raise Exception(f"Failed to start Ollama server: {e}")


def check_model_installed(model_name: str="llama3.2"):
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in result.stdout:
            raise Exception(f"Model {model_name} is not installed.")
        print(f"Model {model_name} is installed.")
    except Exception as e:
        raise Exception(f"Failed to check if model is installed: {e}")


def ollama_on_init_strategy(model_name: str, *args, **kwargs) -> None:
    check_ollama_installed()
    if not check_http_server_running():
        start_ollama_server()
    check_model_installed(model_name=model_name)
