from __future__ import annotations

import io
import select
import subprocess
import sys
import typing as tp


def run_command(command: list[str]) -> str:
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    ) as process:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            copy_process_streams(process, stdout_buffer, stderr_buffer)
        except Exception as e:
            process.kill()
            process.wait()
            raise RuntimeError("Job got killed for an unknown reason.") from e
        stdout = stdout_buffer.getvalue().strip()
        stderr = stderr_buffer.getvalue().strip()
        retcode = process.wait()
        if stderr and retcode:
            print(stderr, file=sys.stderr)
        if retcode:
            subprocess_error = subprocess.CalledProcessError(
                retcode, process.args, output=stdout, stderr=stderr
            )
            raise RuntimeError(stderr) from subprocess_error
    return stdout


def copy_process_streams(
    process: subprocess.Popen,
    stdout: io.StringIO,
    stderr: io.StringIO,
    verbose: bool = False,
):
    """
    Reads the given process stdout/stderr and write them to StringIO objects.
    Make sure that there is no deadlock because of pipe congestion.
    If `verbose` the process stdout/stderr are also copying to the interpreter stdout/stderr.
    """

    def raw(stream: tp.Optional[tp.IO[bytes]]) -> tp.IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: tp.Dict[int, tp.Tuple[tp.IO[bytes], io.StringIO, tp.IO[str]]] = {
        p_stdout.fileno(): (p_stdout, stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())
    poller = select.poll()
    for fd in stream_by_fd:
        poller.register(fd, select.POLLIN | select.POLLPRI)
    while fds:
        # `poll` syscall will wait until one of the registered file descriptors has content.
        ready = poller.poll()
        for fd, _ in ready:
            p_stream, string, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2**16)
            if not raw_buf:
                fds.remove(fd)
                poller.unregister(fd)
                continue
            buf = raw_buf.decode()
            string.write(buf)
            string.flush()
            if verbose:
                std.write(buf)
                std.flush()
