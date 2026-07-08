import os
import logging
import urllib.request
import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

# ── Absolute paths relative to this file ──────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VGG_PATH    = os.path.join(BASE_DIR, 'vgg_normalised.pth')
DEC_PATH    = os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth')
UPLOAD_DIR  = os.path.join(BASE_DIR, 'static', 'uploads')

HF_BASE = 'https://huggingface.co/Bunny6397/AI-NST/resolve/main'

def download_if_missing(path, url):
    if not os.path.exists(path):
        logger.info(f'Downloading {os.path.basename(path)} ...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
        logger.info(f'Downloaded {os.path.basename(path)}')

# Download models at startup if not present (handles Render ephemeral FS)
download_if_missing(VGG_PATH, f'{HF_BASE}/vgg_normalised.pth')
download_if_missing(DEC_PATH, f'{HF_BASE}/decoder_final.pth')

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Load models ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

encoder = VGGEncoder(VGG_PATH).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(DEC_PATH, map_location=device, weights_only=False))
encoder.eval()
decoder.eval()
logger.info('Models loaded successfully')

# ── Form ───────────────────────────────────────────────────────────────────────
class UploadForm(FlaskForm):
    content      = FileField('Content Image')
    style        = FileField('Style Image')
    content_path = HiddenField()
    style_path   = HiddenField()
    alpha        = FloatField('Alpha', default=1.0)
    submit       = SubmitField('Transfer Style')

# ── Helpers ────────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def run_style_transfer(content_img, style_img, alpha):
    tf = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    c = tf(content_img).unsqueeze(0).to(device)
    s = tf(style_img).unsqueeze(0).to(device)
    with torch.no_grad():
        cf = encoder(c, is_test=True)
        sf = encoder(s, is_test=True)
        out = alpha * adaptive_instance_normalization(cf, sf) + (1 - alpha) * cf
        result = decoder(out)
    return result

def save_tensor_image(tensor, path):
    img = tensor.cpu().squeeze(0).clamp(0, 1)
    transforms.ToPILImage()(img).save(path, optimize=True, quality=85)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = content_filename = style_filename = error = None

    if request.method == 'POST':
        if form.validate_on_submit():
            # Save content image
            if form.content.data and form.content.data.filename:
                if allowed_file(form.content.data.filename):
                    content_filename = secure_filename(form.content.data.filename)
                    form.content.data.save(os.path.join(UPLOAD_DIR, content_filename))
                    form.content_path.data = content_filename
            else:
                content_filename = form.content_path.data

            # Save style image
            if form.style.data and form.style.data.filename:
                if allowed_file(form.style.data.filename):
                    style_filename = secure_filename(form.style.data.filename)
                    form.style.data.save(os.path.join(UPLOAD_DIR, style_filename))
                    form.style_path.data = style_filename
            else:
                style_filename = form.style_path.data

            if not content_filename:
                error = 'Please upload a content image.'
            elif not style_filename:
                error = 'Please upload a style image.'
            else:
                try:
                    content_img = Image.open(os.path.join(UPLOAD_DIR, content_filename)).convert('RGB')
                    style_img   = Image.open(os.path.join(UPLOAD_DIR, style_filename)).convert('RGB')
                    alpha       = float(form.alpha.data)
                    result      = run_style_transfer(content_img, style_img, alpha)
                    result_filename = 'stylized_' + content_filename
                    save_tensor_image(result, os.path.join(UPLOAD_DIR, result_filename))
                    result_image = result_filename
                    logger.info(f'Style transfer done: {result_filename}')
                except Exception as e:
                    logger.error(f'Style transfer error: {e}', exc_info=True)
                    error = str(e)
        else:
            error = 'Form submission failed. Please try again.'

    return render_template('index.html', form=form,
                           result_image=result_image,
                           content_image=content_filename,
                           style_image=style_filename,
                           error=error)

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 8080, app, use_reloader=True, use_debugger=True)
