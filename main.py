"""
This script provides a PyQt5-based GUI application for annotating images with bounding boxes.
It supports manual labeling (with user-defined labels) and auto-labeling (where the last drawn box
is automatically assigned a label). The application can also optionally crop and save regions of interest.
"""

import sys
import os
import cv2
import json
import numpy as np
from PIL import Image, ExifTags
from glob import glob
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QFileDialog, QLabel
from PyQt5.QtWidgets import QDesktopWidget, QMessageBox, QCheckBox
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QColor, QPen, QFont
from PyQt5.QtCore import QRect, QPoint

class MyApp(QMainWindow):
    """
    Main application window that hosts the MainWidget and a status bar.
    Manages the application’s primary UI components and geometry.
    """
    def __init__(self):
        """Initialize the main window and setup UI."""
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Set up the main UI for the application:
          - Create a MainWidget for the central widget.
          - Configure a status bar and informational labels.
          - Define window size and properties.
        """
        mainWidget = MainWidget(self)
        self.setCentralWidget(mainWidget)

        # Create a status bar and associated labels
        statusbar = self.statusBar()
        self.setStatusBar(statusbar)
        self.fileName = QLabel('Ready')    # Label to display the current image name
        self.cursorPos = QLabel('      ')  # Label to display the mouse cursor position
        self.imageSize = QLabel('      ')  # Label to display the loaded image size (WxH)
        self.autoLabel = QLabel('Manual Label')  
        self.progress = QLabel('                 ')  # Placeholder label to display progress

        # Container widget for the status bar elements
        widget = QWidget(self)
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(self.fileName)
        widget.layout().addStretch(1)
        widget.layout().addWidget(self.imageSize)
        widget.layout().addWidget(self.cursorPos)
        widget.layout().addStretch(1)
        widget.layout().addWidget(self.autoLabel)
        widget.layout().addStretch(2)
        widget.layout().addWidget(self.progress)
        statusbar.addWidget(widget, 1)

        # Configure main window geometry
        self.setGeometry(50, 50, 1200, 800)
        self.setWindowTitle('im2trainData')
        self.show()

    def fitSize(self):
        """
        Adjust the main window size to fit the layout size hint.
        This keeps the UI compact based on its content.
        """
        self.setFixedSize(self.layout().sizeHint())


class ImageWidget(QWidget):
    """
    A QWidget subclass to display and annotate an image with bounding boxes.
    Users can draw rectangles to label regions of interest, remove boxes,
    and automatically label or manually label those boxes.
    """
    def __init__(self, parent, key_cfg):
        """
        Initialize the ImageWidget.

        Args:
            parent (MyApp): The parent QMainWindow instance.
            key_cfg (list): Configuration for label keys (list of strings).
        """
        super(ImageWidget, self).__init__(parent)
        self.parent = parent          # Reference to parent window
        self.results = []             # Stores bounding boxes and label indices
        self.setMouseTracking(True)   # Enable tracking for the mouse position
        self.key_config = key_cfg     # List of label strings
        self.screen_height = QDesktopWidget().screenGeometry().height()
        self.last_idx = 0             # Stores the last label index

        self.initUI()
        
    def initUI(self):
        """
        Setup the widget’s UI elements.
        - Initialize with a placeholder image.
        - Prepare the layout for displaying the image.
        """
        self.pixmap = QPixmap('start.png')                # Default image when the widget starts
        self.label_img = QLabel()
        self.label_img.setObjectName("image")

        # Keep a copy of the current pixmap for drawing operations
        self.pixmapOriginal = QPixmap.copy(self.pixmap)

        self.drawing = False
        self.lastPoint = QPoint()

        # Layout to hold the image label
        hbox = QHBoxLayout(self.label_img)
        self.setLayout(hbox)

    def paintEvent(self, event):
        """
        Paint event callback to draw the QPixmap onto the widget surface.
        This is automatically called whenever the widget needs to be redrawn.
        """
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)

    def mousePressEvent(self, event):
        """
        Mouse press event handler:
          - If left-click, prepare for drawing a bounding box.
          - If right-click, attempt to remove a bounding box if clicked inside one.
        """
        if event.button() == Qt.LeftButton:
            # Store a copy of current pixmap before starting a new drawing
            self.prev_pixmap = self.pixmap
            self.drawing = True
            self.lastPoint = event.pos()
        elif event.button() == Qt.RightButton:
            x, y = event.pos().x(), event.pos().y()
            # Check each bounding box to see if the click is inside one; if so, remove that box
            for i, box in enumerate(self.results):
                lx, ly, rx, ry = box[:4]
                if lx <= x <= rx and ly <= y <= ry:
                    self.results.pop(i)
                    self.pixmap = self.drawResultBox()
                    self.update()
                    break
            
    def mouseMoveEvent(self, event):
        """
        Mouse move event handler:
          - Updates the cursor position label in the status bar.
          - If left button is pressed and drawing is active, draws a rectangle in real-time.
        """
        # Update the cursor position in the status bar
        self.parent.cursorPos.setText('({}, {})'.format(event.pos().x(), event.pos().y()))

        # If left button is pressed, update the rectangle on the image
        if event.buttons() and Qt.LeftButton and self.drawing:
            self.pixmap = QPixmap.copy(self.prev_pixmap)
            painter = QPainter(self.pixmap)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            p1_x, p1_y = self.lastPoint.x(), self.lastPoint.y()
            p2_x, p2_y = event.pos().x(), event.pos().y()
            painter.drawRect(min(p1_x, p2_x), 
                             min(p1_y, p2_y), 
                             abs(p1_x - p2_x), 
                             abs(p1_y - p2_y))
            self.update()

    def mouseReleaseEvent(self, event):
        """
        Mouse release event handler:
          - Completes the bounding box drawing.
          - If auto-label is set, immediately appends the current label index.
          - If manual label, waits for user to label the box before proceeding.
        """
        if event.button() == Qt.LeftButton:
            p1_x, p1_y = self.lastPoint.x(), self.lastPoint.y() 
            p2_x, p2_y = event.pos().x(), event.pos().y()
            lx, ly = min(p1_x, p2_x), min(p1_y, p2_y)
            w, h = abs(p1_x - p2_x), abs(p1_y - p2_y)

            # Only process if the box is not a zero-area line
            if (p1_x, p1_y) != (p2_x, p2_y):
                # Check if the last box is labeled or not
                if self.results and (len(self.results[-1]) == 4) and self.parent.autoLabel.text() == 'Manual Label':
                    # If the last box is incomplete in manual mode, prompt the user
                    self.showPopupOk('warning messege', 'Please mark the box you drew.')
                    self.pixmap = self.drawResultBox()
                    self.update()
                elif self.parent.autoLabel.text() == 'Auto Label':
                    # If in auto-label mode, append the current label index automatically
                    self.results.append([lx, ly, lx + w, ly + h, self.last_idx])
                    # Fill empty labels for previously drawn but unlabeled boxes
                    for i, result in enumerate(self.results):
                        if len(result) == 4:
                            self.results[i].append(self.last_idx)
                    self.pixmap = self.drawResultBox()
                    self.update()
                else:
                    # Manual label mode: store just the box, label to be assigned later
                    self.results.append([lx, ly, lx + w, ly + h])
                self.drawing = False

    def showPopupOk(self, title: str, content: str):
        """
        Show a message box popup with a single "Ok" button.

        Args:
            title (str): Title of the message box.
            content (str): Main text content of the popup.
        """
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        if result == QMessageBox.Ok:
            msg.close()

    def drawResultBox(self):
        """
        Redraw the bounding boxes onto a fresh copy of the original pixmap.

        Returns:
            QPixmap: The updated pixmap with all bounding boxes.
        """
        res = QPixmap.copy(self.pixmapOriginal)
        painter = QPainter(res)

        # Use a font for labeling the bounding box text
        font = QFont('mono', 15, 1)
        painter.setFont(font)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

        # Draw each bounding box; if it has a label (5 items), draw that as text in blue
        for box in self.results:
            lx, ly, rx, ry = box[:4]
            painter.drawRect(lx, ly, rx - lx, ry - ly)
            if len(box) == 5:
                painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
                painter.drawText(lx, ly + 15, self.key_config[box[-1]])
                painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        return res

    def setPixmap(self, image_fn):
        """
        Load a new image into the widget from a file, resize it if it exceeds screen height.

        Args:
            image_fn (str): Path to the image file.
        """
        self.pixmap = QPixmap(image_fn)
        self.W, self.H = self.pixmap.width(), self.pixmap.height()

        # Scale down if image is too tall for the screen
        if self.H > self.screen_height * 0.8:
            resize_ratio = (self.screen_height * 0.8) / self.H
            self.W = round(self.W * resize_ratio)
            self.H = round(self.H * resize_ratio)
            self.pixmap = QPixmap.scaled(self.pixmap, 
                                         self.W, 
                                         self.H,
                                         transformMode=Qt.SmoothTransformation)
        
        # Update status bar and fix widget size
        self.parent.imageSize.setText('{}x{}'.format(self.W, self.H))
        self.setFixedSize(self.W, self.H)
        self.pixmapOriginal = QPixmap.copy(self.pixmap)

    def cancelLast(self):
        """
        Remove the most recently added bounding box from the results list, if any.
        Useful for the user to undo the last drawing.
        """
        if self.results:
            self.results.pop()
            self.pixmap = self.drawResultBox()
            self.update()
    
    def getRatio(self):
        """
        Get the current displayed image width and height.

        Returns:
            tuple: (width, height) of the displayed QPixmap.
        """
        return self.W, self.H

    def getResult(self):
        """
        Provide the list of all bounding boxes for the current image.

        Returns:
            list: A list of bounding box definitions, each either [lx, ly, rx, ry] or [lx, ly, rx, ry, idx].
        """
        return self.results

    def resetResult(self):
        """
        Clear any drawn bounding boxes and reset the results list for a fresh image.
        """
        self.results = []

    def markBox(self, idx):
        """
        Assign a label to the most recently drawn bounding box.

        Args:
            idx (int): Label index to assign (the position within key_config).
        """
        self.last_idx = idx
        if self.results:
            if len(self.results[-1]) == 4:
                # If the last box is missing a label, append this new label
                self.results[-1].append(idx)
            elif len(self.results[-1]) == 5:
                # If the last box already had a label, replace it
                self.results[-1][-1] = idx
            else:
                raise ValueError('invalid results')
            self.pixmap = self.drawResultBox()
            self.update()


class MainWidget(QWidget):
    """
    MainWidget manages the user interface for selecting input paths, saving paths,
    toggling crop mode, loading images in a sequence, and coordinating the annotation process.
    """
    def __init__(self, parent):
        """
        Initialize the MainWidget UI and load label configurations.

        Args:
            parent (MyApp): The parent main window instance.
        """
        super(MainWidget, self).__init__(parent)
        self.parent = parent
        self.currentImg = "start.png"

        # Load config from JSON
        config_dict = self.getConfigFromJson('config.json')
        # Gather all keys from the config for labeling; they are named 'key_1' .. 'key_9'
        self.key_config = [
            config_dict['key_'+str(i)] 
            for i in range(1, 10) if config_dict['key_'+str(i)]
        ]
        self.crop_mode = False
        self.save_directory = None

        self.initUI()

    def initUI(self):
        """
        Set up the UI elements in MainWidget:
          - Path selection buttons.
          - Labels for displaying selected paths.
          - Checkbox for toggling crop mode.
          - Navigation buttons (next, cancel).
          - ImageWidget for annotating images.
        """
        # Create and configure UI elements
        inputPathButton = QPushButton('Input Path', self)
        savePathButton = QPushButton('Save Path', self)
        savePathButton.setEnabled(False)  # Disabled until crop mode is active
        okButton = QPushButton('Next', self)
        cancelButton = QPushButton('Cancel', self)
        cropModeCheckBox = QCheckBox("Crop Mode", self)
        inputPathLabel = QLabel('Input Path not selected', self)
        self.savePathLabel = QLabel('Save Path not selected', self)
        self.savePathLabel.setEnabled(False)

        # Our custom widget for displaying/labeling images
        self.label_img = ImageWidget(self.parent, self.key_config)

        # Connect signals to corresponding slots/callbacks
        okButton.clicked.connect(self.setNextImage)
        okButton.setEnabled(False)
        cancelButton.clicked.connect(self.label_img.cancelLast)
        cropModeCheckBox.stateChanged.connect(
            lambda state: self.cropMode(state, savePathButton)
        )
        inputPathButton.clicked.connect(
            lambda: self.registerInputPath(inputPathButton, inputPathLabel, okButton)
        )
        savePathButton.clicked.connect(
            lambda: self.registerSavePath(savePathButton, self.savePathLabel)
        )
        
        # Layout for the first column (two buttons)
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(inputPathButton)
        vbox.addWidget(savePathButton)
        hbox.addLayout(vbox)

        # Layout for the second column (two labels)
        vbox = QVBoxLayout()
        vbox.addWidget(inputPathLabel)
        vbox.addWidget(self.savePathLabel)
        hbox.addLayout(vbox)

        # Spacer to push items to the left
        hbox.addStretch(3)

        # Add the crop-mode checkbox in the row
        hbox.addWidget(cropModeCheckBox)
        hbox.addStretch(1)

        # Add "Next" and "Cancel" buttons
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)

        # Main vertical layout: image on top, then the horizontal button layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.label_img)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def setNextImage(self, img=None):
        """
        Load the next image from the directory, save the existing annotations, and reset for new annotations.
        Handles orientation corrections for certain EXIF flags (rotates image if needed).

        Args:
            img (str): Optional image path for forced loading. If None, proceeds to the next image in the list.
        """
        # If crop mode is active and label indicates "Results" folder, ensure that directory is created
        if self.savePathLabel.text() == 'Results' and self.crop_mode:
            os.makedirs(self.save_directory, exist_ok=True)

        if not img:
            # Write bounding box results for the current image
            res = self.label_img.getResult()
            # If the last box is incomplete (missing label), show a warning
            if res and len(res[-1]) != 5:
                self.label_img.showPopupOk('warning messege', 'please mark the box you drew.')
                return 'Not Marked'
            self.writeResults(res)
            self.label_img.resetResult()

            # Attempt to load the next image from the list; if none remain, display an "end" placeholder
            try:
                self.currentImg = self.imgList.pop(0)
            except Exception:
                self.currentImg = 'end.png'
        else:
            # If a specific image is passed, skip results saving and just load
            self.label_img.resetResult()

        # Correct orientation if needed (some cameras store orientation in EXIF)
        try:
            im = Image.open(self.currentImg)
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break 
            exif = dict(im.getexif().items())
            # If EXIF orientation is 3, 6, or 8, rotate the image
            if exif[orientation] in [3, 6, 8]:
                im = im.transpose(Image.ROTATE_180)
                im.save(self.currentImg)
        except:
            pass

        # Update the UI to show the newly loaded image
        basename = os.path.basename(self.currentImg)
        self.parent.fileName.setText(basename)
        self.parent.progress.setText(str(self.total_imgs - len(self.imgList)) + '/' + str(self.total_imgs))

        self.label_img.setPixmap(self.currentImg)
        self.label_img.update()
        self.parent.fitSize()

    def writeResults(self, res: list):
        """
        Save bounding box annotations to a .txt file in YOLO format and optionally crop regions.

        Args:
            res (list): A list of bounding boxes, each in the form [lx, ly, rx, ry, idx].
        """
        # Skip if no image is loaded yet
        if self.parent.fileName.text() != 'Ready':
            W, H = self.label_img.getRatio()

            # If there are no bounding boxes, create an empty file for YOLO
            if not res:
                open(self.currentImg[:-4]+'.txt', 'a', encoding='utf8').close()

            # Process each bounding box and write YOLO format lines
            for i, elements in enumerate(res):
                lx, ly, rx, ry, idx = elements
                # YOLO format: (class, center_x / w, center_y / h, width / w, height / h)
                yolo_format = [
                    idx,
                    (lx + rx) / 2 / W,      # center x ratio
                    (ly + ry) / 2 / H,      # center y ratio
                    (rx - lx) / W,          # width ratio
                    (ry - ly) / H           # height ratio
                ]
                with open(self.currentImg[:-4]+'.txt', 'a', encoding='utf8') as resultFile:
                    resultFile.write(' '.join([str(x) for x in yolo_format])+'\n')

                # Crop mode: create separate images for each bounding box
                if self.crop_mode:
                    img = cv2.imread(self.currentImg)
                    # If the image path has non-ASCII characters (e.g., Korean), load with NumPy + cv2.imdecode
                    if img is None:
                        n = np.fromfile(self.currentImg, np.uint8)
                        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
                    oh, ow = img.shape[:2]

                    # Convert ratio-based coords back to absolute pixel coords
                    w, h = round(yolo_format[3] * ow), round(yolo_format[4] * oh)
                    x = round(yolo_format[1] * ow - w / 2)
                    y = round(yolo_format[2] * oh - h / 2)

                    # Crop and save the ROI
                    crop_img = img[y : y + h, x : x + w]
                    basename = os.path.basename(self.currentImg)
                    filename = basename[:-4] + '-{}-{}.jpg'.format(self.key_config[idx], i)

                    # Convert from OpenCV BGR to RGB for PIL saving
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    crop_img = Image.fromarray(crop_img)
                    # Save the cropped image in the designated directory
                    crop_img.save(os.path.join(self.save_directory, filename), dpi=(300,300))

    def registerSavePath(self, savePathButton, label):
        """
        Open a dialog to select a directory for saving cropped images.
        Update the label to reflect the chosen path.

        Args:
            savePathButton (QPushButton): Button that triggers save path selection.
            label (QLabel): Label to display the chosen save path name.
        """
        savePathButton.toggle()
        self.save_directory = str(QFileDialog.getExistingDirectory(self, "Select Save Directory"))
        basename = os.path.basename(self.save_directory)
        if basename:
            label.setText(basename + '/')
        else:
            print("Output Path not selected")
            self.save_directory = None

    def registerInputPath(self, inputPathButton, inputPathLabel, okButton):
        """
        Open a dialog to select the input directory. Load all images (jpg, png) into a list,
        skipping any images that already have a .txt file. Also sets a default "Results" subfolder
        for saving if none is chosen.

        Args:
            inputPathButton (QPushButton): Button that triggers the input path selection.
            inputPathLabel (QLabel): Label to display the chosen input path name.
            okButton (QPushButton): Button for proceeding to the next image (enables after selection).
        """
        inputPathButton.toggle()
        directory = str(QFileDialog.getExistingDirectory(self, "Select Input Directory"))
        basename = os.path.basename(directory)

        if not basename:
            print("Input Path not selected")
            return -1
        
        # Get all .jpg and .png files from the selected directory
        types = ('*.jpg', '*.png')
        self.imgList = []
        for t in types:
            self.imgList.extend(glob(directory + '/' + t))
        self.total_imgs = len(self.imgList)

        # Skip images that already have .txt files (already labeled)
        to_skip = []
        for imgPath in self.imgList:
            if os.path.exists(imgPath[:-4] + '.txt'):
                to_skip.append(imgPath)
        for skip in to_skip:
            self.imgList.remove(skip)

        inputPathLabel.setText(basename + '/')
        okButton.setEnabled(True)

        # If no save directory is set, default to a "Results" subfolder in the input directory
        if self.save_directory is None or self.savePathLabel.text() == 'Results':
            self.savePathLabel.setText('Results')
            self.save_directory = os.path.join(directory, 'Results')

    def getConfigFromJson(self, json_file):
        """
        Load labeling configuration from a JSON file.
        Expects key_1 ... key_9 to exist in the file.

        Args:
            json_file (str): The path to the JSON configuration file.

        Returns:
            dict: A dictionary of configurations.
        """
        with open(json_file, 'r') as config_file:
            try:
                config_dict = json.load(config_file)
                return config_dict
            except ValueError:
                print("INVALID JSON file format.. Please provide a valid JSON file")
                exit(-1)

    def cropMode(self, state, savePathButton):
        """
        Toggle crop mode on or off. When crop mode is enabled, saving images is allowed.

        Args:
            state (int): The state of the QCheckBox (checked or not).
            savePathButton (QPushButton): Button to enable/disable for selecting the save path.
        """
        if state == Qt.Checked:
            self.crop_mode = True
            savePathButton.setEnabled(True)
        else:
            self.crop_mode = False
            savePathButton.setEnabled(False)

    def keyPressEvent(self, e):
        """
        Capture keyboard events for assigning labels, undoing boxes, navigating images,
        and toggling auto-label vs. manual-label modes.

        Args:
            e (QKeyEvent): The key press event.
        """
        config_len = len(self.key_config)
        # Number keys 1-9 to label the last bounding box
        for i, key_n in enumerate(range(49, 58), 1):
            if e.key() == key_n and config_len >= i:
                self.label_img.markBox(i - 1)
                break

        # Esc to cancel the last bounding box
        if e.key() == Qt.Key_Escape:
            self.label_img.cancelLast()
        # 'E' to confirm and move to the next image
        elif e.key() == Qt.Key_E:
            self.setNextImage()
        # 'Q' to reset all bounding boxes on the current image
        elif e.key() == Qt.Key_Q:
            self.label_img.resetResult()
            self.label_img.pixmap = self.label_img.drawResultBox()
            self.label_img.update()
        # 'A' to toggle Auto Label and Manual Label
        elif e.key() == Qt.Key_A:
            if self.parent.autoLabel.text() == 'Auto Label':
                self.parent.autoLabel.setText('Manual Label')
            else:
                self.parent.autoLabel.setText('Auto Label')


if __name__ == '__main__':
    """
    Entry point for the application.
    Initializes the QApplication, constructs and displays the main window.
    """
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
