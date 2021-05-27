import subprocess

if __name__ == "__main__":
    subprocess.call("pkill -9 -f run_app.py", shell=True)
    subprocess.call("pkill -9 -f run_bot.py", shell=True)
