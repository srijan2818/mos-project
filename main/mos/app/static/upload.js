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

  window.mosUpload = {
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
