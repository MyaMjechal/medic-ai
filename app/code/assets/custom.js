// capture Enter vs Shift+Enter in the therapy textarea
document.addEventListener('DOMContentLoaded', function() {
    const ta = document.getElementById('therapy-input');
    const btn = document.getElementById('therapy-send');
    if (ta && btn) {
      ta.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          btn.click();
        }
      });
    }
  });
  