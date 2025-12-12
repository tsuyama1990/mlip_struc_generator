
import subprocess
import logging
import platform
import os
import time
import threading

logger = logging.getLogger(__name__)

class KeepAwake:
    """
    Context manager to prevent Windows Sleep Mode from WSL.
    
    It spawns a background PowerShell process that calls SetThreadExecutionState
    with ES_SYSTEM_REQUIRED | ES_CONTINUOUS.
    """
    
    def __init__(self):
        self.proc = None
        self._is_wsl = False
        
    def _check_wsl(self):
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    return True
        except FileNotFoundError:
            pass
        return False

    def __enter__(self):
        self._is_wsl = self._check_wsl()
        
        if self._is_wsl:
            logger.info("KeepAwake: WSL detected. Preventing Windows Sleep Mode...")
            # PowerShell command to set execution state and stay alive
            # 0x80000003 = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            # 0x80000001 = ES_CONTINUOUS | ES_SYSTEM_REQUIRED (No display)
            # Users usually prefer display on to monitor progress, so 0x80000003 is safe.
            ps_script = """
            $code = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
            $type = Add-Type -MemberDefinition $code -Name "Win32" -Namespace Win32 -PassThru
            $type::SetThreadExecutionState(0x80000003)
            Start-Sleep -Seconds 86400
            """
            
            # We must run powershell.exe from WSL path.
            # Usually strict path is /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
            # But 'powershell.exe' usually works if PATH is set.
            try:
                self.proc = subprocess.Popen(
                    ["powershell.exe", "-Command", ps_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.debug(f"KeepAwake: Started PowerShell process (PID {self.proc.pid})")
            except FileNotFoundError:
                logger.warning("KeepAwake: powershell.exe not found. Cannot prevent sleep.")
            except Exception as e:
                logger.warning(f"KeepAwake: Failed to start blocking process: {e}")
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
                logger.debug("KeepAwake: Terminated blocking process.")
            except Exception as e:
                logger.warning(f"KeepAwake: Error processing shutdown: {e}")
            
            # Reset state just in case (though process termination should be enough)
            # We can't easily call SetThreadExecutionState(ES_CONTINUOUS) back to normal 
            # without another script, but terminating the holding process usually releases the request.
