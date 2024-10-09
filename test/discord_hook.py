import datetime
import requests

discord_url = "https://discord.com/api/webhooks/1293069085835530281/iBC65WPVKUw1t5AglmnUb6iG7gao2B5dBY-IWQhGQ7TrzKI3fjpHijrBXSH8-MGcbrSp"

#디스코드 채널로 메세지 전송
def discord_send_message(text):
    now = datetime.datetime.now()
    message = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {str(text)}"}
    requests.post(discord_url, data=message)
    print(message)
    
discord_send_message("health-check")
