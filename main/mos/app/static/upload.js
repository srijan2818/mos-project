// Browser-side file pickers for the MoS app. Vue templates can't reach `document`,
// `FileReader` etc., so the inline handlers call these helpers instead:
//   change="window.mosUpload.single($event, {data: 'cv_image_b64'})"
(function () {
  const set = (key, value) => key && window.trame.state.set(key, value);

  const readDataURL = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error('could not read ' + file.name));
    reader.readAsDataURL(file);
  });

  // Downscale so the longest side is <= maxSide, as JPEG (keeps synced state small).
  const resized = async (file, maxSide) => {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    if (bitmap.close) bitmap.close();
    return canvas.toDataURL('image/jpeg', 0.88);
  };

  // Live camera dialog (getUserMedia). Needs a secure origin: localhost or https.
  const openCamera = async (keys, fallbackInputId) => {
    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
      set(keys.status, 'No camera access here (needs localhost or https); choose an image file instead.');
      document.getElementById(fallbackInputId).click();
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(
        { video: { facingMode: 'environment', width: { ideal: 1920 } }, audio: false });
    } catch (error) {
      set(keys.status, `Camera unavailable (${error.name}); choose an image file instead.`);
      document.getElementById(fallbackInputId).click();
      return;
    }
    const overlay = document.createElement('div');
    overlay.className = 'mos-camera';
    overlay.innerHTML = '<video autoplay playsinline muted></video>' +
      '<div class="mos-camera-bar"><button data-act="snap">Capture</button>' +
      '<button data-act="cancel">Cancel</button></div>';
    document.body.appendChild(overlay);
    const video = overlay.querySelector('video');
    video.srcObject = stream;
    const close = () => { stream.getTracks().forEach((t) => t.stop()); overlay.remove(); };
    overlay.querySelector('[data-act=cancel]').onclick = close;
    overlay.querySelector('[data-act=snap]').onclick = () => {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      set(keys.data, canvas.toDataURL('image/png'));
      set(keys.status, `Camera photo captured (${canvas.width} x ${canvas.height} px). Press Run CV Model.`);
      close();
    };
  };

  const MAX_FRAMES = 40;

  // Frames from the live camera while the user walks around the part (no video file: a
  // recorded WebM often has no duration, which breaks seeking).
  const recordFrames = async (keys, maxSide) => {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(
        { video: { facingMode: 'environment', width: { ideal: 1920 } }, audio: false });
    } catch (error) {
      set(keys.status, `Camera unavailable (${error.name}); upload a video file instead.`);
      return;
    }
    const overlay = document.createElement('div');
    overlay.className = 'mos-camera';
    overlay.innerHTML = '<video autoplay playsinline muted></video>' +
      '<div class="mos-camera-count">Start straight above the part, then circle it ~25° above the table.</div>' +
      '<div class="mos-camera-bar"><button data-act="rec">Start</button>' +
      '<button data-act="stop" disabled>Stop &amp; use</button><button data-act="cancel">Cancel</button></div>';
    document.body.appendChild(overlay);
    const video = overlay.querySelector('video');
    const count = overlay.querySelector('.mos-camera-count');
    video.srcObject = stream;
    const frames = [];
    let timer = null;
    const close = () => { clearInterval(timer); stream.getTracks().forEach((t) => t.stop()); overlay.remove(); };
    overlay.querySelector('[data-act=cancel]').onclick = close;
    overlay.querySelector('[data-act=rec]').onclick = (ev) => {
      ev.target.disabled = true;
      overlay.querySelector('[data-act=stop]').disabled = false;
      timer = setInterval(() => {
        if (frames.length >= MAX_FRAMES) return;
        frames.push(grab(video, maxSide));
        count.textContent = `${frames.length} frames (max ${MAX_FRAMES})`;
      }, 400);
    };
    overlay.querySelector('[data-act=stop]').onclick = () => {
      close();
      set(keys.frames, frames);
      set(keys.status, `${frames.length} frames captured.`);
    };
  };

  const grab = (video, maxSide) => {
    const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.9);
  };

  // Evenly spaced frames from an uploaded video file (phone MP4 etc.).
  const fileFrames = async (event, keys, maxSide) => {
    const input = event.target;
    const file = input.files && input.files[0];
    input.value = '';
    if (!file) return;
    const video = document.createElement('video');
    video.muted = true;
    video.src = URL.createObjectURL(file);
    try {
      await new Promise((ok, bad) => { video.onloadedmetadata = ok; video.onerror = () => bad(new Error('this browser cannot decode the video')); });
      if (!isFinite(video.duration) || video.duration <= 0) throw new Error('the video has no duration');
      const n = Math.min(MAX_FRAMES, Math.max(8, Math.round(video.duration / 0.4)));
      const frames = [];
      for (let i = 0; i < n; i++) {
        video.currentTime = (i + 0.5) * video.duration / n;
        await new Promise((ok) => { video.onseeked = ok; });
        frames.push(grab(video, maxSide));
        set(keys.status, `Extracting frames… ${i + 1}/${n}`);
      }
      set(keys.frames, frames);
      set(keys.status, `${frames.length} frames from ${file.name}.`);
    } catch (error) {
      set(keys.status, 'Could not read the video: ' + error.message);
    } finally {
      URL.revokeObjectURL(video.src);
    }
  };

  window.mosUpload = {
    // Record frames from the camera while orbiting the part; frames -> state[keys.frames].
    recordFrames(keys, maxSide) { recordFrames(keys, maxSide || 1280); },

    // Frames from an uploaded video file.
    videoFrames(event, keys, maxSide) { fileFrames(event, keys, maxSide || 1280); },

    // Open the hidden <input type=file> with this id.
    pick(id) { document.getElementById(id).click(); },

    // Live camera capture; falls back to the file input when there is no camera.
    camera(keys, fallbackInputId) { openCamera(keys, fallbackInputId); },

    // One image -> data URL in state[keys.data]; optional keys.name / keys.status.
    async single(event, keys, maxMB) {
      const input = event.target;
      const file = input.files && input.files[0];
      input.value = '';
      if (!file) return;
      if (maxMB && file.size > maxMB * 1024 * 1024) {
        set(keys.status, `Image is larger than ${maxMB} MB. Resize it and try again.`);
        return;
      }
      try {
        set(keys.data, await readDataURL(file));
        set(keys.name, file.name);
        set(keys.status, `Image ready: ${file.name}.`);
      } catch (error) {
        set(keys.status, 'Could not read the selected image: ' + error.message);
      }
    },

    // Several photos -> [{name, data}] in cv_mv_images (each resized to <= 1600 px).
    async multiview(event) {
      const input = event.target;
      const files = Array.from(input.files || []);
      input.value = '';
      if (!files.length) return;
      try {
        const items = await Promise.all(
          files.map(async (file) => ({ name: file.name, data: await resized(file, 1600) })));
        set('cv_mv_images', items);
        set('cv_mv_file_names', items.map((item) => item.name));
        set('cv_mv_status', `${items.length} photos ready. Build the capture dataset to validate and save them.`);
      } catch (error) {
        set('cv_mv_status', 'Image upload error: ' + error.message);
      }
    },
  };
})();
