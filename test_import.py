import sys
if __name__ == '__main__':
    try:
        import app.main
    except BaseException as e:
        import traceback
        with open("error.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
