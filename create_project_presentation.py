from pathlib import Path
from xml.sax.saxutils import escape
import struct
import zipfile


BASE = Path(__file__).resolve().parent
REPORT = BASE / "report"
IMAGES = REPORT / "images"
OUT = REPORT / "steam_game_success_predictor_presentation.pptx"

SLIDE_W = 12192000
SLIDE_H = 6858000


def png_size(path):
    with open(path, "rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def emu(inches):
    return int(inches * 914400)


def text_box(shape_id, x, y, w, h, text, size=28, bold=False, color="172033"):
    runs = []
    for line in text.split("\n"):
        runs.append(
            f"""
            <a:p>
              <a:r>
                <a:rPr lang="en-US" sz="{size * 100}" b="{1 if bold else 0}">
                  <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                </a:rPr>
                <a:t>{escape(line)}</a:t>
              </a:r>
            </a:p>"""
        )
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{x}" y="{y}"/>
          <a:ext cx="{w}" cy="{h}"/>
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square"/>
        <a:lstStyle/>
        {''.join(runs)}
      </p:txBody>
    </p:sp>"""


def bullets_box(shape_id, x, y, w, h, bullets, size=22, color="172033"):
    paragraphs = []
    for item in bullets:
        paragraphs.append(
            f"""
            <a:p>
              <a:pPr marL="342900" indent="-171450">
                <a:buChar char="•"/>
              </a:pPr>
              <a:r>
                <a:rPr lang="en-US" sz="{size * 100}">
                  <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                </a:rPr>
                <a:t>{escape(item)}</a:t>
              </a:r>
            </a:p>"""
        )
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="Bullets {shape_id}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{x}" y="{y}"/>
          <a:ext cx="{w}" cy="{h}"/>
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square"/>
        <a:lstStyle/>
        {''.join(paragraphs)}
      </p:txBody>
    </p:sp>"""


def image_shape(shape_id, rel_id, x, y, w, h, name):
    return f"""
    <p:pic>
      <p:nvPicPr>
        <p:cNvPr id="{shape_id}" name="{escape(name)}"/>
        <p:cNvPicPr/>
        <p:nvPr/>
      </p:nvPicPr>
      <p:blipFill>
        <a:blip r:embed="{rel_id}"/>
        <a:stretch><a:fillRect/></a:stretch>
      </p:blipFill>
      <p:spPr>
        <a:xfrm>
          <a:off x="{x}" y="{y}"/>
          <a:ext cx="{w}" cy="{h}"/>
        </a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
      </p:spPr>
    </p:pic>"""


def fit_image(path, x, y, max_w, max_h):
    px_w, px_h = png_size(path)
    scale = min(max_w / px_w, max_h / px_h)
    return x, y, int(px_w * scale), int(px_h * scale)


slides = [
    {
        "title": "Steam Game Success Predictor",
        "subtitle": "KNN, SVM, Random Forest and Deep Learning",
        "bullets": [
            "Binary classification project based on Steam game data",
            "Four trained models compared in one Streamlit deployment",
            "Includes evaluation, cross-validation and documentation",
        ],
        "image": "streamlit_main_form.png",
    },
    {
        "title": "Project Goal",
        "bullets": [
            "Predict whether a Steam game is successful or not successful",
            "Use numerical features such as reviews, playtime, price and CCU",
            "Compare classical ML models with a neural network",
            "Deploy all models in a single interactive application",
        ],
        "image": "ml_workflow_diagram.png",
    },
    {
        "title": "Dataset and Target Variable",
        "bullets": [
            "Target variable: success",
            "Success is based on rating ratio and concurrent user activity",
            "Leakage columns are removed from improved models",
        ],
        "image": "target_distribution.png",
    },
    {
        "title": "Preprocessing in Jupyter",
        "bullets": [
            "Create rating_ratio and success",
            "Remove text columns and leakage-related columns",
            "Prepare final feature matrix for model training",
        ],
        "image": "notebook_target_preprocessing.png",
    },
    {
        "title": "Implemented Models",
        "bullets": [
            "KNN: distance-based baseline model",
            "SVM: RBF kernel with balanced class weights",
            "Random Forest: tree ensemble for tabular data",
            "Deep Learning: dense neural network with sigmoid output",
        ],
        "image": "model_accuracy_comparison.png",
    },
    {
        "title": "Classical ML Notebook Code",
        "bullets": [
            "KNN and SVM use StandardScaler",
            "Random Forest uses class_weight='balanced'",
            "Models are evaluated with accuracy, classification report and confusion matrix",
        ],
        "images": ["notebook_knn_code.png", "notebook_random_forest_code.png"],
    },
    {
        "title": "Deep Learning Model",
        "bullets": [
            "Neural network built with TensorFlow/Keras",
            "Dense layers with ReLU activations",
            "Sigmoid output for binary classification",
        ],
        "image": "notebook_deep_learning_code.png",
    },
    {
        "title": "Deep Learning Training Curves",
        "bullets": [
            "Accuracy increases during training",
            "Validation loss suggests possible later overfitting",
            "Final test accuracy is approximately 0.763",
        ],
        "images": ["deep_learning_training_1.png", "deep_learning_training_2.png"],
    },
    {
        "title": "Cross-Validation",
        "bullets": [
            "Added to KNN, SVM and Random Forest notebooks",
            "Mean accuracy shows average model performance",
            "Standard deviation shows model stability between folds",
            "KNN and SVM use Pipeline to avoid scaling leakage",
        ],
        "image": "code_cross_validation.png",
    },
    {
        "title": "Combined Streamlit Deployment",
        "bullets": [
            "Root app.py loads all four saved models",
            "One input form is shared by all models",
            "Results are shown side by side with probability or score",
        ],
        "image": "streamlit_successful_prediction.png",
    },
    {
        "title": "Deployment Code",
        "bullets": [
            "Models and scalers are loaded from project folders",
            "SVM uses decision_function because it was trained without probability=True",
            "Preset buttons fill examples for successful and unsuccessful games",
        ],
        "images": ["code_combined_deployment.png", "code_prediction_logic.png"],
    },
    {
        "title": "Limitations and Future Work",
        "bullets": [
            "Success definition is an approximation, not real revenue",
            "SVM probability should be calibrated by retraining with probability=True",
            "Future work: GridSearchCV, ROC-AUC, more features, online deployment",
            "Project demonstrates the full ML pipeline from data to deployment",
        ],
        "image": "streamlit_unsuccessful_prediction.png",
    },
]


def slide_xml(index, slide):
    title = text_box(2, emu(0.55), emu(0.25), emu(12.2), emu(0.6), slide["title"], 30, True)
    shapes = [title]

    if "subtitle" in slide:
        shapes.append(text_box(3, emu(0.6), emu(0.95), emu(6.0), emu(0.4), slide["subtitle"], 18, False, "44546A"))
        bullet_y = emu(1.45)
    else:
        bullet_y = emu(1.05)

    shapes.append(bullets_box(4, emu(0.55), bullet_y, emu(4.1), emu(4.8), slide["bullets"], 18))

    rels = []
    media = []
    next_id = 5

    if "image" in slide:
        image_name = slide["image"]
        image_path = IMAGES / image_name
        x, y, w, h = fit_image(image_path, emu(4.75), emu(1.15), emu(7.75), emu(5.65))
        shapes.append(image_shape(next_id, "rId2", x, y, w, h, image_name))
        rels.append(("rId2", f"../media/{image_name}"))
        media.append(image_name)
    elif "images" in slide:
        for offset, image_name in enumerate(slide["images"]):
            image_path = IMAGES / image_name
            y = emu(1.05 + offset * 2.85)
            x, y, w, h = fit_image(image_path, emu(4.75), y, emu(7.75), emu(2.55))
            rel_id = f"rId{2 + offset}"
            shapes.append(image_shape(next_id + offset, rel_id, x, y, w, h, image_name))
            rels.append((rel_id, f"../media/{image_name}"))
            media.append(image_name)

    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgPr>
        <a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      {''.join(shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""

    rel_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  {''.join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>' for rid, target in rels)}
</Relationships>"""

    return xml, rel_xml, media


def write_package():
    media_names = []
    slide_parts = []

    for index, slide in enumerate(slides, start=1):
        xml, rel_xml, media = slide_xml(index, slide)
        slide_parts.append((index, xml, rel_xml))
        media_names.extend(media)

    media_names = list(dict.fromkeys(media_names))

    content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  {''.join(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i, _, _ in slide_parts)}
</Types>"""

    presentation_rels = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i, _, _ in slide_parts
    )
    master_rid = len(slide_parts) + 1
    theme_rid = len(slide_parts) + 2
    presentation_rels += f'\n<Relationship Id="rId{master_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    presentation_rels += f'\n<Relationship Id="rId{theme_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>'

    presentation = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId{master_rid}"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
    {''.join(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i, _, _ in slide_parts)}
  </p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    ppt_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {presentation_rels}
</Relationships>"""

    slide_master = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""

    slide_master_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""

    slide_layout = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

    slide_layout_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""

    theme = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Steam ML Theme">
  <a:themeElements>
    <a:clrScheme name="Office">
      <a:dk1><a:srgbClr val="172033"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="44546A"/></a:dk2><a:lt2><a:srgbClr val="F5F7FB"/></a:lt2>
      <a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="22C55E"/></a:accent2>
      <a:accent3><a:srgbClr val="F97316"/></a:accent3><a:accent4><a:srgbClr val="A78BFA"/></a:accent4>
      <a:accent5><a:srgbClr val="38BDF8"/></a:accent5><a:accent6><a:srgbClr val="EF4444"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>
  </a:themeElements>
</a:theme>"""

    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Steam Game Success Predictor</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
</cp:coreProperties>"""

    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft PowerPoint</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>12</Slides>
</Properties>"""

    if OUT.exists():
        OUT.unlink()

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("ppt/presentation.xml", presentation)
        z.writestr("ppt/_rels/presentation.xml.rels", ppt_rels)
        z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels)
        z.writestr("ppt/theme/theme1.xml", theme)

        for index, xml, rel_xml in slide_parts:
            z.writestr(f"ppt/slides/slide{index}.xml", xml)
            z.writestr(f"ppt/slides/_rels/slide{index}.xml.rels", rel_xml)

        for image_name in media_names:
            z.write(IMAGES / image_name, f"ppt/media/{image_name}")

    print(OUT)


write_package()
