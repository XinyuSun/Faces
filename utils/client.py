import pysftp
import json
import os
from pathlib import Path

class sftpServer(object):
    def __init__(self):
        cfg = json.load(open("config/server.json",'r'))
        self.sftp = pysftp.Connection(cfg["host"], username=cfg["user"], password="4092")
        self.root = "Projects/Face/container"
        self.sftp.cwd(self.root)
    
    def send(self, target:str, remote:str)->int:
        if not os.path.exists(target):
            return -1
        else:
            self.sftp.put(target, remote)

    def recv(self, target:str, local:str)->int:
        if not self.sftp.exists(target):
            return -1
        else:
            self.sftp.get(target, local)

    def sync(self, path:str)->int:
        print(self.sftp.pwd)
        if not os.path.exists(path):
            if not self.sftp.exists(path):
                return -1
            else:
                self.sftp.get_r(path, path, preserve_mtime=True)
        elif not self.sftp.exists(path):
            self.sftp.mkdir(path)
            self.sftp.put_r(path, path, confirm=True, preserve_mtime=True)
        else:
            remote_files = self.sftp.listdir(path)
            local_files = os.listdir(path)
            all_files = set(remote_files + local_files)
            remote_files_missing = list(all_files - set(remote_files))
            local_files_missing = list(all_files - set(local_files))
            print(f'missing files: {remote_files_missing} and {local_files_missing}')
            for rfm in remote_files_missing:
                target = os.path.join(path, rfm)
                self.send(target, Path.as_posix(Path(target)))
            print(f"upload {len(remote_files_missing)} files")

            for lfm in local_files_missing:
                target = os.path.join(path, lfm)
                print(self.sftp.pwd)
                self.recv(Path.as_posix(Path(target)), target)
                print(target)
            print(f"download {len(local_files_missing)} files")

if __name__ == "__main__":
    server = sftpServer()
    #print(server.syncLocal('data', '.'))
    print(server.sync('data'))