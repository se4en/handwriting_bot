from app import flask_app


if __name__ == "__main__":
    # start app
    flask_app.run(debug=True, host="0.0.0.0")
