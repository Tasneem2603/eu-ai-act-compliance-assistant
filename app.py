import os

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from chatbot.llm import ask_llm
from chatbot.pdf_reader import load_pdf, get_active_document, load_default_corpus

from config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH
)


# Flask Configuration


app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Create documents folder automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preload the EU AI Act grounding corpus (Task 3 grounding requirement)
# so the assistant has something to retrieve from even before anyone
# uploads a file. Safe to call even if the file isn't present.
load_default_corpus()



# Helper Function


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )



# Home Page


@app.route("/")
def home():

    return render_template("index.html")



# Chat API


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    question = data.get("message", "")

    if question == "":

        return jsonify({

            "response": "Please enter a question."

        })

    answer = ask_llm(question)

    return jsonify({

        "response": answer

    })



# Upload API


@app.route("/upload", methods=["POST"])
def upload():

    if "file" not in request.files:

        return jsonify({

            "success": False,

            "message": "No file selected."

        })

    file = request.files["file"]

    if file.filename == "":

        return jsonify({

            "success": False,

            "message": "No filename."

        })

    if not allowed_file(file.filename):

        return jsonify({

            "success": False,

            "message": "Unsupported file type."

        })

    filename = secure_filename(file.filename)

    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )

    file.save(filepath)

    # Load + index the uploaded file (PDF or TXT) into the vector store
    load_pdf(filepath)

    return jsonify({

        "success": True,

        "message": f"{filename} uploaded successfully."

    })



# Knowledge Base API


@app.route("/documents", methods=["GET"])
def get_documents():

    documents = []

    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):

        documents.append(filename)

    return jsonify({

        "documents": documents

    })



# Active Document API


@app.route("/active-document", methods=["GET"])
def active_document():

    return jsonify(get_active_document())



# Run Application


if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
