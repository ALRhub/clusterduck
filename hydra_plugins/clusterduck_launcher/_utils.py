import subprocess


def run_command(command: list[str]) -> str:
    try:
        # Run the command, capture output, and check for errors automatically
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Automatically decodes bytes to strings
            check=True,  # Automatically raises CalledProcessError if retcode != 0
            shell=False,  # Keeps your shell-injection security intact
        )
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        # If the command failed, e.stderr contains the error message
        raise RuntimeError(f"Command failed: {e.stderr.strip()}") from e
