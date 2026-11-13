"""Shared desktop palette and native Qt widget styling.

Assets are deliberately bundled with the application: no external font, icon
service, network request or additional runtime dependency is required.
"""

from pathlib import Path
import json


# Refactoring UI §1.5 / §2.1 / §3.2 / §4.1 / §5.6:
# primary = current observation / current-stage action; secondary = evidence;
# tertiary = model diagnostics. Keep the existing forest palette and flat depth.
# New ramp stops are generated from the existing bases; text has three roles.
_ASSETS = Path(__file__).resolve().parent / "assets"
PALETTES = json.loads((_ASSETS / "design-palettes.json").read_text(encoding="utf-8"))
SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32, "section": 48}
TYPE = {"meta": 12, "body": 16, "subheading": 20, "heading": 26, "stat": 34, "hero": 44}
RADIUS = {"control": 8, "card": 12}


COLORS = {
    "background": "#F4F7F6",
    "surface": "#FFFFFF",
    "border": "#E3EBE7",
    "sidebar": "#173A36",
    "primary": "#1F7665",
    "primary_hover": "#195F52",
    "text": "#182C28",
    "secondary_text": PALETTES["neutral"]["700"]["hex"],
    "tertiary_text": PALETTES["neutral"]["600"]["hex"],
    "muted": PALETTES["neutral"]["600"]["hex"],
    "success": PALETTES["primary"]["700"]["hex"],
    "warning": PALETTES["warning"]["600"]["hex"],
    "warning_surface": "#FBF2E3",
    "soft_surface": "#F8FAF8",
    "tinted_surface": "#EDF5F0",
    "tinted_text": PALETTES["primary"]["800"]["hex"],
    "chart_visual": PALETTES["neutral"]["600"]["hex"],
    "chart_audio": PALETTES["warning"]["500"]["hex"],
    "chart_joint": "#1F7665",
    "breathing_text": PALETTES["primary"]["950"]["hex"],
}

_CHEVRON = (_ASSETS / "chevron-down.svg").as_posix()
_CHECK = (_ASSETS / "check.svg").as_posix()

APP_STYLE = """
QWidget {
    background: transparent;
    color: #182C28;
    font-family: 'Microsoft YaHei UI', 'Microsoft YaHei', 'Segoe UI';
    font-size: 12px;
}
QMainWindow, QWidget#root, QWidget#central {
    background: #F4F7F6;
}
QLabel { background: transparent; border: 0; }
QFrame { border: 0; }
QFrame#card, QGroupBox {
    background: #FFFFFF;
    border: 1px solid #E3EBE7;
    border-radius: 12px;
}
QFrame#sessionPanel {
    background: #FFFFFF;
    border: 1px solid #E3EBE7;
    border-radius: 12px;
}
QFrame#sidebar, QWidget#sidebar {
    background: #173A36;
    border: 0;
}
QLabel#brandMark {
    background: #28534B;
    color: #D7F2DE;
    border: 1px solid #3A655B;
    border-radius: 13px;
    font-size: 22px;
    font-weight: 600;
}
QLabel#brandTitle { color: #FFFFFF; font-size: 18px; font-weight: 600; }
QLabel#brandSubtitle { color: #94B5AA; font-size: 11px; }
QLabel#sidebarCaption {
    color: #91B0A5;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
}
QLabel#sidebarNote { color: #ACCCC0; font-size: 11px; }
QLabel#headerTitle, QLabel#title {
    color: #182C28;
    font-size: 23px;
    font-weight: 600;
}
QLabel#headerSubtitle { color: #73847B; font-size: 12px; }
QLabel#cardTitle, QLabel#sectionTitle {
    color: #244039;
    font-size: 14px;
    font-weight: 600;
}
QLabel#muted { color: #65786C; font-size: 12px; }
QLabel#smallMuted { color: #728276; font-size: 11px; }
QLabel#eyebrow {
    color: #73847B;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}
QLabel#scoreValue { color: #1F7665; font-size: 32px; font-weight: 600; }
QLabel#scoreUnit { color: #84948C; font-size: 12px; }
QLabel#sessionBadge, QLabel#badge {
    background: #EAF2ED;
    color: #517167;
    border: 1px solid #DCE8E0;
    border-radius: 12px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 600;
}
QLabel#successPill {
    background: #E9F4EC;
    color: #2B7B63;
    border-radius: 10px;
    padding: 4px 9px;
    font-size: 11px;
}
QLabel#warningPill {
    background: #FBF2E3;
    color: #946A30;
    border-radius: 10px;
    padding: 4px 9px;
    font-size: 11px;
}
QLabel#notice, QLabel#infoStrip, QFrame#infoStrip {
    background: #EDF5F0;
    color: #527569;
    border: 1px solid #E1EDE5;
    border-radius: 9px;
    padding: 9px 12px;
    font-size: 11px;
}
QFrame#stepBar, QLabel#stepBar {
    background: #F0F5F2;
    color: #628074;
    border: 0;
    border-radius: 9px;
    padding: 9px 12px;
}
QLabel#emptyTitle { color: #3F6156; font-size: 14px; font-weight: 600; }
QLabel#emptyBody { color: #84948C; font-size: 11px; }

QPushButton {
    background: #FFFFFF;
    color: #3D5A50;
    border: 1px solid #DBE5DF;
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 500;
}
QPushButton:hover { background: #F0F6F2; border-color: #B7CEC0; }
QPushButton:pressed { background: #E4EEE7; border-color: #9CB8A7; }
QPushButton:focus { border-color: #1F7665; }
QPushButton:disabled {
    background: #F3F6F4;
    color: #A2AFA7;
    border-color: #E7EDE9;
}
QPushButton#primary {
    background: #1F7665;
    color: #FFFFFF;
    border-color: #1F7665;
    font-weight: 600;
}
QPushButton#primary:hover { background: #195F52; border-color: #195F52; }
QPushButton#primary:pressed { background: #124D43; border-color: #124D43; }
QPushButton#primary:focus { border-color: #72B9A4; }
QPushButton#primary:disabled {
    background: #B7CDC1;
    border-color: #B7CDC1;
    color: #F4F8F5;
}
QPushButton#secondary { background: #EDF5F0; color: #2E6E59; border-color: #D3E5D9; }
QPushButton#secondary:hover { background: #E0EEE5; border-color: #ABCAB8; }
QPushButton#secondary:disabled { color: #A2AFA7; background: #F3F6F4; border-color: #E7EDE9; }
QPushButton#danger { color: #9E5B50; border-color: #E8D7D2; background: #FFFFFF; }
QPushButton#danger:hover { color: #89493F; border-color: #D7AAA0; background: #FCF3F0; }
QPushButton#danger:disabled { color: #B7ABA7; border-color: #E9E3DF; background: #F7F6F4; }
QPushButton#ghost { color: #628074; background: transparent; border-color: transparent; }
QPushButton#ghost:hover { color: #1F7665; background: #EAF2ED; }
QPushButton#ghost:disabled { color: #ABB8B0; background: transparent; }
QPushButton#navButton {
    color: #AFCCC1;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 12px 15px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
}
QPushButton#navButton:hover { background: #20483F; color: #FFFFFF; }
QPushButton#navButton:checked {
    background: #2D584B;
    color: #F0F9EF;
    border-color: #365F52;
    font-weight: 600;
}
QPushButton#navButton:focus { border-color: #6E9B87; }

QLineEdit, QComboBox, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {
    background: #FFFFFF;
    color: #314C41;
    border: 1px solid #DBE5DF;
    border-radius: 7px;
    padding: 7px 9px;
    selection-background-color: #D5EADB;
    selection-color: #182C28;
}
QLineEdit:hover, QComboBox:hover, QPlainTextEdit:hover, QTextEdit:hover {
    border-color: #B3CCBD;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border-color: #1F7665;
}
QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {
    background: #F5F7F5;
    color: #A0AEA4;
    border-color: #E5ECE7;
}
QComboBox { min-height: 20px; padding-right: 32px; }
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border: 0;
    background: transparent;
}
QComboBox::down-arrow { image: url('__CHEVRON__'); width: 12px; height: 12px; }
QComboBox QAbstractItemView {
    background: #FFFFFF;
    color: #314C41;
    selection-background-color: #E4F1E9;
    selection-color: #1F7665;
    border: 1px solid #D3E1D8;
    outline: 0;
    padding: 4px;
}
QCheckBox { background: transparent; color: #526D5F; spacing: 8px; }
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    background: #FFFFFF;
    border: 1px solid #BDD0C2;
    border-radius: 4px;
}
QCheckBox::indicator:hover { border-color: #1F7665; background: #F2F8F3; }
QCheckBox::indicator:checked {
    background: #1F7665;
    border-color: #1F7665;
    image: url('__CHECK__');
}
QCheckBox::indicator:checked:hover { background: #195F52; border-color: #195F52; }
QCheckBox::indicator:disabled { border-color: #DDE6DF; background: #F1F5F2; }
QCheckBox::indicator:checked:disabled { background: #ADC8B7; border-color: #ADC8B7; }
QCheckBox:disabled { color: #A0AEA4; }
QCheckBox:focus { color: #1F7665; }

QProgressBar {
    border: 0;
    border-radius: 4px;
    background: #ECF2EE;
    text-align: center;
    color: #476C59;
    font-size: 10px;
}
QProgressBar::chunk { background: #A9D3BA; border-radius: 4px; }
QTabWidget { background: transparent; }
QTabWidget::pane { border: 0; background: transparent; }
QTabBar::tab {
    background: transparent;
    color: #73847B;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: 10px 18px;
}
QTabBar::tab:selected { color: #1F7665; border-bottom-color: #1F7665; font-weight: 600; }
QTabBar::tab:hover { background: #EBF2ED; }
QTableWidget, QTableView {
    background: #FFFFFF;
    alternate-background-color: #F8FAF8;
    color: #476153;
    border: 1px solid #E3EBE7;
    border-radius: 8px;
    gridline-color: #ECF1ED;
    selection-background-color: #E7F2EA;
    selection-color: #244E3C;
    outline: 0;
}
QTableWidget::item { padding: 8px; border: 0; }
QTableWidget::item:selected { background: #E7F2EA; color: #244E3C; }
QHeaderView { background: #F1F6F2; border: 0; padding: 0; }
QHeaderView::section {
    background: #F1F6F2;
    color: #688273;
    border: 0;
    border-bottom: 1px solid #E3EBE7;
    padding: 10px;
    font-size: 11px;
    font-weight: 600;
}
QTableCornerButton::section { background: #F1F6F2; border: 0; }
QScrollArea { background: transparent; border: 0; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 3px 0; }
QScrollBar::handle:vertical { background: #CDDBD1; border-radius: 4px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: #A4BFAE; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 0 3px; }
QScrollBar::handle:horizontal { background: #CDDBD1; border-radius: 4px; min-width: 28px; }
QScrollBar::handle:horizontal:hover { background: #A4BFAE; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QSplitter::handle { background: transparent; }
QStatusBar {
    background: #F4F7F6;
    color: #84948C;
    border-top: 1px solid #E6EDE8;
    font-size: 10px;
    padding: 3px 12px;
}
QStatusBar::item { border: 0; }
QToolTip {
    background: #24483E;
    color: #F4F9F3;
    border: 1px solid #365D4D;
    border-radius: 5px;
    padding: 6px 9px;
    font-size: 11px;
}
QMessageBox, QDialog { background: #F8FAF8; }
""".replace("__CHEVRON__", _CHEVRON).replace("__CHECK__", _CHECK)


# §4.1: promote the existing typography to one named pixel scale, including
# QPainter labels. §5.6: semantic labels must remain readable on their surfaces.
_FONT_ROLES = {10: "meta", 11: "meta", 12: "body", 13: "body", 14: "body",
               18: "subheading", 22: "heading", 23: "heading", 32: "hero"}
import re as _re
APP_STYLE = _re.sub(r"font-size:\s*(\d+)px", lambda m: f"font-size: {TYPE[_FONT_ROLES.get(int(m[1]), 'body')]}px", APP_STYLE)
_TEXT_REPLACEMENTS = {
    "#73847B": COLORS["tertiary_text"], "#84948C": COLORS["tertiary_text"],
    "#728276": COLORS["tertiary_text"], "#65786C": COLORS["secondary_text"],
    "#688273": COLORS["secondary_text"], "#628074": COLORS["tinted_text"],
    "#527569": COLORS["tinted_text"], "#476C59": COLORS["secondary_text"],
    "#2B7B63": COLORS["success"], "#946A30": COLORS["warning"],
}
for _old, _new in _TEXT_REPLACEMENTS.items():
    APP_STYLE = APP_STYLE.replace(_old, _new)

APP_STYLE += f"""
QLabel#muted {{ color:{COLORS['secondary_text']}; font-size:{TYPE['meta']}px; }}
QLabel#smallMuted, QLabel#eyebrow, QLabel#scoreUnit {{ color:{COLORS['tertiary_text']}; font-size:{TYPE['meta']}px; }}
QLabel#headerSubtitle {{ color:{COLORS['secondary_text']}; font-size:{TYPE['body']}px; }}
QLabel#cardTitle, QLabel#sectionTitle {{ font-size:{TYPE['body']}px; }}
QLabel#dominant {{ color:{COLORS['text']}; font-size:{TYPE['subheading']}px; font-weight:600; }}
QLabel#activityName {{ color:{COLORS['primary']}; font-size:{TYPE['heading']}px; font-weight:600; }}
QLabel#countdown {{ color:{COLORS['text']}; font-size:{TYPE['stat']}px; font-weight:600; }}
QLabel#scoreValue {{ color:{COLORS['primary']}; font-size:{TYPE['hero']}px; font-weight:600; }}
QLabel#statValue {{ color:{COLORS['text']}; font-size:{TYPE['stat']}px; font-weight:600; }}
QLineEdit, QComboBox, QPlainTextEdit {{ placeholder-text-color:{COLORS['tertiary_text']}; }}
QLabel#infoStrip, QFrame#infoStrip {{ color:{COLORS['tinted_text']}; }}
QLabel#bodyNote {{ color:{COLORS['secondary_text']}; font-size:{TYPE['body']}px; }}
QPushButton {{ color:{COLORS['secondary_text']}; }}
QLabel#cardTitle, QLabel#sectionTitle {{ color:{COLORS['text']}; }}
QLineEdit, QComboBox, QPlainTextEdit, QTableWidget {{ color:{COLORS['text']}; }}
QProgressBar {{ color:{COLORS['text']}; }}
QLabel#savedFeedback {{ color:{COLORS['tinted_text']}; background:{COLORS['tinted_surface']};
    border-radius:{RADIUS['control']}px; padding:{SPACE['lg']}px; font-size:{TYPE['body']}px; }}
"""
