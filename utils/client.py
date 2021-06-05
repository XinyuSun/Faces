import pysftp
import json
import os
import paramiko
from base64 import decodebytes

class sftpServer(object):
    def __init__(self):
        cfg = json.load(open("config/server.json",'r'))
        key = paramiko.RSAKey(data=decodebytes(bytes(cfg["key"], encoding='utf-8')))
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys.add(cfg["host"], cfg["decode"], key)
        self.sftp = pysftp.Connection(cfg["host"], username=cfg["user"], password="4092", cnopts=cnopts)
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
                self.send(target, target)
            print(f"updated {len(remote_files_missing)} files")

            for lfm in local_files_missing:
                target = os.path.join(path, lfm)
                self.recv(target, target)
            print(f"updated {len(local_files_missing)} files")

if __name__ == "__main__":
    server = sftpServer()
    #print(server.syncLocal('data', '.'))
    print(server.sync('data'))