# python module to set font attributes for text printed to sys.stdout
#
# Usage:
#     To print boldface text to the terminal:
#
#        print boldface + "Hello World!" + end_format

end_formatting = end_format = reset = '\033[0m'
boldface = bold = bf = '\033[1m'
underline = '\033[4m'

k = black = '\033[30m'
r = red = '\033[31m'
g = green = '\033[32m'
y = yellow = '\033[33m'
b = blue = '\033[34m'
m = magenta = '\033[35m'
c = cyan = '\033[36m'
w = white = '\033[37m'

# Background colors
# 40    Black
# 41    Red
# 42    Green
# 43    Yellow
# 44    Blue
# 45    Magenta
# 46    Cyan
# 47    White
