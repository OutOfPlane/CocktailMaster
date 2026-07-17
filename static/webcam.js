/* Reusable "take a photo" modal, shared by every page with an image upload.
 *
 *   openWebcam(file => { ... })
 *
 * Opens a fullscreen camera preview; on capture it hands the callback a JPEG
 * File (the same shape a file <input> or a drag-drop would produce), so callers
 * can reuse their existing upload path unchanged. The camera is always released
 * on close.
 *
 * Note: browsers only grant camera access in a secure context -- https or
 * http://localhost. Reaching the app over a plain-http LAN address (e.g. a
 * tablet hitting the host's IP) will be blocked by the browser, not by this
 * code; the error surfaces in the modal.
 */
(function () {
    let stream = null;
    let facing = "environment";  // prefer the rear camera on phones/tablets
    let onDone = null;
    let overlay, video, errBox;

    function build() {
        overlay = document.createElement("div");
        overlay.className = "webcam-overlay";
        overlay.innerHTML = `
            <div class="webcam-modal">
                <video class="webcam-video" autoplay playsinline muted></video>
                <p class="webcam-error" hidden></p>
                <div class="webcam-controls">
                    <button type="button" class="webcam-btn webcam-cancel">✕ Abbrechen</button>
                    <button type="button" class="webcam-btn webcam-flip">🔄 Kamera</button>
                    <button type="button" class="webcam-btn webcam-shoot">📸 Aufnehmen</button>
                </div>
            </div>`;
        document.body.appendChild(overlay);
        video = overlay.querySelector(".webcam-video");
        errBox = overlay.querySelector(".webcam-error");
        overlay.querySelector(".webcam-cancel").onclick = close;
        overlay.querySelector(".webcam-flip").onclick = flip;
        overlay.querySelector(".webcam-shoot").onclick = shoot;
        // Click the backdrop (outside the modal) to dismiss.
        overlay.addEventListener("click", e => { if (e.target === overlay) close(); });
        // Esc also dismisses.
        document.addEventListener("keydown", onKey);
    }

    function onKey(e) { if (e.key === "Escape") close(); }

    async function start() {
        stopStream();
        errBox.hidden = true;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: facing }, audio: false,
            });
            video.srcObject = stream;
        } catch (e) {
            errBox.textContent = "Kein Kamerazugriff: " + (e.message || e.name || e);
            errBox.hidden = false;
        }
    }

    function stopStream() {
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
        if (video) video.srcObject = null;
    }

    function flip() {
        facing = facing === "environment" ? "user" : "environment";
        start();
    }

    function shoot() {
        if (!stream) return;
        const w = video.videoWidth, h = video.videoHeight;
        if (!w || !h) return;  // stream not ready yet
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        canvas.getContext("2d").drawImage(video, 0, 0, w, h);
        canvas.toBlob(blob => {
            if (!blob) return;
            const file = new File([blob], "webcam-" + Date.now() + ".jpg", { type: "image/jpeg" });
            const cb = onDone;
            close();
            if (cb) cb(file);
        }, "image/jpeg", 0.9);
    }

    function close() {
        stopStream();
        document.removeEventListener("keydown", onKey);
        if (overlay) overlay.remove();
        overlay = null;
        onDone = null;
    }

    window.openWebcam = function (callback) {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Dieser Browser unterstützt keine Kamera.");
            return;
        }
        onDone = callback;
        build();
        start();
    };
})();
