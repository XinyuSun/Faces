import sys
from PyQt5.QtWidgets import QApplication
from utils.ui import uiWindow


if __name__ == '__main__':
    # main()
    app = QApplication(sys.argv)
    window = uiWindow()
    sys.exit(app.exec_())