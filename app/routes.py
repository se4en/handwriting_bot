from flask import (
    request,
    render_template,
    flash,
    redirect,
    jsonify,
    make_response,
    url_for,
    session,
)
from app import flask_app
import os
import sys
import stat
import base64
import uuid
import numpy as np
from app.priming import generate_handwriting
from app.xml_parser import svg_xml_parser, path_to_stroke, path_string_to_stroke


# sys.path.append("../")
from utils import plot_stroke


@flask_app.route("/", methods=["GET"])
def draw():
    if "id" in session:
        id = session["id"]
        print("uuid: ", id)
    return render_template("draw.html", title="Write")

@flask_app.route("/upload_style", methods=["GET", "POST"])
def submit_style_data():
    data = request.get_json()
    path = data["path"]
    text = data["text"]
    if path == "":
        return jsonify(
            dict({"redirect": url_for("draw"), "message": "Please enter some style"})
        )

    id = str(uuid.uuid4())
    session["id"] = id
    tmp_dir = os.path.join(flask_app.root_path, "static", "uploads", id)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    os.chmod(tmp_dir, 0o777)
    print(tmp_dir)
    # user agent info
    user_agent = request.user_agent
    print(user_agent.string)
    print(user_agent.platform)
    phones = ["android", "iphone"]
    down_sample = True

    if user_agent.platform in phones:
        down_sample = False

    text_path = os.path.join(tmp_dir, "inpText.txt")
    print(text_path)
    with open(text_path, "w") as f:
        f.write(text)
    f.close()

    stroke = path_string_to_stroke(
        path, str_len=len(list(text)), down_sample=down_sample
    )

    # here we need use file name for this user !
    save_path = os.path.join(tmp_dir, "style.npy")
    np.save(save_path, stroke, allow_pickle=True)
    print(save_path)

    # plot the sequence
    #
    # here we need use file name for this user also !
    plot_stroke(stroke.astype(np.float32), os.path.join(tmp_dir, "original.png"))

    return jsonify(dict({"redirect": url_for("generate"), "message": ""}))
