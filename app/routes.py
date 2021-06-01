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
from app.xml_parser import path_string_to_stroke


# sys.path.append("../")
from utils import plot_stroke


@flask_app.route("/", methods=["GET", "POST"])
def draw():
    if "id" in session:
        id = session["id"]
        print("uuid: ", id)
    name = request.args.get('name')
    return render_template("draw.html", title="Write", name=name)


@flask_app.route("/upload_style", methods=["GET", "POST"])
def submit_style_data():
    data = request.get_json()
    path = data["path"]
    text = data["text"]
    user_name = data["name"]
    if path == "":
        return jsonify(
            dict({"redirect": url_for("draw"), "message": "Please enter some style"})
        )

    data_dir = "app/user_data"
    # here we need use file name for this user !!!
    #user_name = "user_1"

    # user agent info
    user_agent = request.user_agent
    print(user_agent.string)
    print(user_agent.platform)
    phones = ["android", "iphone"]
    down_sample = True

    if user_agent.platform in phones:
        down_sample = False

    # save user text
    text_path = os.path.join(data_dir, user_name + ".txt")
    print(text_path)
    with open(text_path, "w") as f:
        f.write(text)
    f.close()

    stroke = path_string_to_stroke(
        path, str_len=len(list(text)), down_sample=down_sample
    )

    # save user stroke
    save_path = os.path.join(data_dir, user_name + ".npy")
    np.save(save_path, stroke, allow_pickle=True)
    print(save_path)

    # save user plot
    plot_stroke(stroke.astype(np.float32), os.path.join(data_dir, user_name + ".png"))

    return jsonify(dict({"redirect": url_for("draw"), "message": ""}))
