from flask import Flask, request, send_from_directory
import subprocess
import os
import openai

app = Flask(__name__, static_folder='web')

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/install', methods=['POST'])
def install_packages():
    packages = request.json.get('packages', '')
    if not packages:
        return 'No packages provided', 400
    cmd = ['apt-get', 'update']
    out = subprocess.run(cmd, capture_output=True, text=True)
    install_cmd = ['apt-get', 'install', '-y'] + packages.split()
    proc = subprocess.run(install_cmd, capture_output=True, text=True)
    return out.stdout + out.stderr + proc.stdout + proc.stderr

@app.route('/remove', methods=['POST'])
def remove_packages():
    packages = request.json.get('packages', '')
    if not packages:
        return 'No packages provided', 400
    cmd = ['apt-get', 'remove', '-y'] + packages.split()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout + proc.stderr

@app.route('/run', methods=['POST'])
def run_command():
    command = request.json.get('command', '')
    if not command:
        return 'No command provided', 400
    proc = subprocess.run(command, shell=True, capture_output=True, text=True)
    return proc.stdout + proc.stderr

@app.route('/chat', methods=['POST'])
def chat_model():
    prompt = request.json.get('prompt', '')
    if not prompt:
        return 'No prompt provided', 400
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
