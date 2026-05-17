import time
from main import main

if __name__ == "__main__":
      target_time = "14:30"

      while True:
          formatted_time = time.strftime("%H:%M", time.localtime())
          print("awaiting", formatted_time)

          if formatted_time == target_time:
              main()
              break

          time.sleep(60)
