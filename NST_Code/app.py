import os
import gc
import uuid
import logging
import threading
import torch
from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, url_for
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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VGG_PATH   = os.path.join(BASE_DIR, 'vgg_normalised.pth')
DEC_PATH   = os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Load models once at startup ────────────────────────────────────────────────
device = torch.device('cpu')
logger.info('Loading models...')
encoder = VGGEncoder(VGG_PATH).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(DEC_PATH, map_location=device, weights_only=False))
encoder.eval()
decoder.eval()
logger.info('Models ready')

# ── In-memory job store ────────────────────────────────────────────────────────
# { job_id: {'status': 'pending'|'done'|'error', 'result': filename, 'error': msg } }
jobs = {}
jobs_lock = threading.Lock()

# ── Form ───────────────────────────────────────────────────────────────────────
class UploadForm(FlaskForm):
    content      = FileField('Content Image')
    style        = FileField('Style Image')
    content_path = HiddenField()
    style_path   = HiddenField()
    alpha        = FloatField('Alpha', default=1.0)
    submit       = SubmitField('Transfer Style')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ── Background inference ───────────────────────────────────────────────────────
def do_transfer(job_id, content_path, style_path, alpha):
    try:
        tf = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
        c_img = Image.open(content_path).convert('RGB')
        s_img = Image.open(style_path).convert('RGB')
        c = tf(c_img).unsqueeze(0)
        s = tf(s_img).unsqueeze(0)
        with torch.no_grad():
            cf = encoder(c, is_test=True)
            sf = encoder(s, is_test=True)
            out = alpha * adaptive_instance_normalization(cf, sf) + (1 - alpha) * cf
            result_tensor = decoder(out)
        # Save result
        result_filename = 'stylized_' + os.path.basename(content_path)
        result_path = os.path.join(UPLOAD_DIR, result_filename)
        img = result_tensor.cpu().squeeze(0).clamp(0, 1)
        transforms.ToPILImage()(img).save(result_path, optimize=True, quality=85)
        # Free memory
        del c, s, cf, sf, out, result_tensor, img
        gc.collect()
        with jobs_lock:
            jobs[job_id] = {'status': 'done', 'result': result_filename}
        logger.info(f'Job {job_id} done: {result_filename}')
    except Exception as e:
        logger.error(f'Job {job_id} error: {e}', exc_info=True)
        with jobs_lock:
            jobs[job_id] = {'status': 'error', 'error': str(e)}

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = content_filename = style_filename = error = job_id = None

    if request.method == 'POST':
        if form.validate_on_submit():
            if form.content.data and form.content.data.filename:
                if allowed_file(form.content.data.filename):
                    content_filename = secure_filename(form.content.data.filename)
                    form.content.data.save(os.path.join(UPLOAD_DIR, content_filename))
            else:
                content_filename = form.content_path.data

            if form.style.data and form.style.data.filename:
                if allowed_file(form.style.data.filename):
                    style_filename = secure_filename(form.style.data.filename)
                    form.style.data.save(os.path.join(UPLOAD_DIR, style_filename))
            else:
                style_filename = form.style_path.data

            if not content_filename:
                error = 'Please upload a content image.'
            elif not style_filename:
                error = 'Please upload a style image.'
            else:
                # Start background job — return immediately, no timeout
                job_id = str(uuid.uuid4())
                with jobs_lock:
                    jobs[job_id] = {'status': 'pending'}
                t = threading.Thread(
                    target=do_transfer,
                    args=(job_id,
                          os.path.join(UPLOAD_DIR, content_filename),
                          os.path.join(UPLOAD_DIR, style_filename),
                          float(form.alpha.data)),
                    daemon=True
                )
                t.start()
        else:
            error = 'Form submission failed. Please try again.'

    return render_template('index.html', form=form,
                           result_image=result_image,
                           content_image=content_filename,
                           style_image=style_filename,
                           error=error,
                           job_id=job_id)

@app.route('/status/<job_id>')
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'})
    return jsonify(job)

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 8080, app, use_reloader=True, use_debugger=True)
