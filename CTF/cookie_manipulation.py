import subprocess
import re
import requests

# セッションを接続するための準備
url = ''
s= requests.Session()

# 一度ポストし、クッキーの情報を得る
r = s.post(url, data = {"submit":""})
data = s.cookies["ship"]
signature = s.cookies["signature"]

# hashpump を subprocess で呼ぶ
args = {}
args["data"] = data
args["signature"] = signature
args["key"] = 21
args["append"] = ",10"

cmd = "hashpump -s {signature} -k {key} -d {data} "
cmd += "-a {append}"
cmd = cmd.format(**args)

proc = subprocess.Popen(cmd.strip().split(" "), stdout=subprocess.PIPE)
out, err = proc.communicate()

# 得られた cookie を url エンコードにする
crack_signature, crack_data = out.decode("utf-8").strip().split("\n")
crack_data = crack_data.replace("\\x","%")

# cookie を変更して再接続
s.cookies.clear()
setargs = {"domain":"","path":""}
s.cookies.set("ship",crack_data,**setargs)
s.cookies.set("signature",crack_signature,**setargs)
r = s.get(url)
