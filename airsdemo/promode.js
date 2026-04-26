/* ============================================================
   pro-mode-sync.js  —  Syncs professional mode state on load
   Drop this into every page that has an agent-image.
   The toggle BUTTON only lives on landpage.html.
   All other pages just listen and apply the stored state.
   ============================================================ */

(function () {
    const KEY = 'airs_professional_mode';

    // Apply state immediately to avoid flash
    function applyState(enabled) {
        if (enabled) {
            document.body.classList.add('professional-mode');
        } else {
            document.body.classList.remove('professional-mode');
        }
    }

    // Apply on load from localStorage
    applyState(localStorage.getItem(KEY) === 'true');

    // Listen for changes from other tabs (e.g. toggle on landpage)
    if (typeof BroadcastChannel !== 'undefined') {
        const ch = new BroadcastChannel('airs_pro_mode');
        ch.onmessage = function (e) {
            localStorage.setItem(KEY, e.data.proMode);
            applyState(e.data.proMode);
        };
    }
})();