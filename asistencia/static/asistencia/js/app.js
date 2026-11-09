/* VisionAsistencia — lógica de la interfaz (sin dependencias externas) */
const VA = (() => {
  const csrf = () => document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';

  async function postJSON(url, datos = {}) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf() },
      body: JSON.stringify(datos),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  async function getJSON(url) {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const $ = (id) => document.getElementById(id);

  /* ---------------------------------------------------------- en vivo */
  const COLORES_EMOCION = {
    neutral: '#94a3b8', feliz: '#eab308', sorpresa: '#3b82f6', tristeza: '#1e40af',
    enojo: '#dc2626', asco: '#16a34a', miedo: '#9333ea', desprecio: '#c2410c',
  };
  const NOMBRES_EMOCION = {
    neutral: 'Neutral', feliz: 'Feliz', sorpresa: 'Sorpresa', tristeza: 'Tristeza',
    enojo: 'Enojo', asco: 'Asco', miedo: 'Miedo', desprecio: 'Desprecio',
  };

  function barraEmocion(emocion, nombre, porcentaje) {
    const p = Math.max(0, Math.min(100, porcentaje));
    const color = COLORES_EMOCION[emocion] || '#64748b';
    return `<li><span class="emo-nombre">${esc(nombre)}</span>
      <span class="emo-pista"><span class="emo-relleno" style="width:${p}%;background:${color}"></span></span>
      <span class="emo-valor">${Math.round(p)}%</span></li>`;
  }

  function paginaEnVivo(urls) {
    const video = $('video');
    const insignia = $('insignia');
    const btnDetener = $('btn-detener');
    const btnIniciar = $('btn-iniciar');
    let pausado = false;

    btnDetener.addEventListener('click', async () => {
      pausado = true;
      video.src = '';
      await postJSON(urls.detener).catch(() => {});
      insignia.textContent = '❚❚ EN PAUSA';
      insignia.classList.add('pausa');
      btnDetener.hidden = true; btnIniciar.hidden = false;
    });
    btnIniciar.addEventListener('click', () => {
      pausado = false;
      video.src = video.dataset.src + '?t=' + Date.now();
      insignia.textContent = '● EN VIVO';
      insignia.classList.remove('pausa');
      btnDetener.hidden = false; btnIniciar.hidden = true;
    });
    $('btn-reiniciar').addEventListener('click', async () => {
      await postJSON(urls.reiniciar).catch(() => {});
      $('resumen').innerHTML = '<li class="vacio">Esperando rostros…</li>';
    });
    // Si el stream se corta (servidor reiniciado, etc.) se reintenta.
    video.addEventListener('error', () => {
      if (!pausado) setTimeout(() => { video.src = video.dataset.src + '?t=' + Date.now(); }, 2000);
    });

    async function actualizarEstado() {
      if (pausado) return;
      try {
        const e = await getJSON(urls.estado);
        $('m-fps').textContent = e.fps ?? '–';
        $('m-rostros').textContent = e.rostros ?? '–';
        $('m-emocion').textContent = e.emocion_principal || '–';
        $('m-luz').textContent = e.iluminacion ? `${e.iluminacion} (${Math.round(e.brillo)})` : '–';
        $('m-camara').textContent = e.camara_ok ? 'conectada' : (e.en_ejecucion ? 'reintentando…' : 'iniciando…');
        if (e.modo_emociones) {
          $('m-modo').innerHTML = e.modo_emociones.startsWith('FER')
            ? `<span class="estado-ok">● Modelo: ${esc(e.modo_emociones)}</span>`
            : `<span class="estado-mal">● Modelo: ${esc(e.modo_emociones)}: sin la red FER+ solo distingue feliz/neutral</span>`;
        }

        $('en-cuadro').innerHTML = (e.caras || []).map((c) => {
          const top = Object.entries(c.probabilidades).slice(0, 3);
          return `<li class="emo-cara" style="border-left-color:${COLORES_EMOCION[c.emocion] || '#cbd5e1'}">
            <b>Rostro ${c.numero}: ${esc(c.nombre)}</b> · ${Math.round(c.confianza)}%
            <ul class="emo-barras">${top.map(([emo, p]) => barraEmocion(emo, NOMBRES_EMOCION[emo] || emo, p)).join('')}</ul></li>`;
        }).join('');

        const resumen = (e.resumen || []).filter((r) => r.cuadros > 0).sort((a, b) => b.cuadros - a.cuadros);
        if (resumen.length) {
          $('resumen').innerHTML = resumen.map((r) => barraEmocion(r.emocion, r.nombre, r.porcentaje)).join('');
        }
      } catch (_) { /* el servidor puede estar reiniciando */ }
    }

    actualizarEstado();
    setInterval(actualizarEstado, 700);
    window.addEventListener('beforeunload', () => { video.src = ''; });
  }

  /* ------------------------------------------------- registro de rostro */
  function paginaRegistro() {
    const btn = $('btn-registro');
    if (!btn) return;
    const btnCancelar = $('btn-cancelar-registro');
    const marco = $('registro-video');
    const img = $('video-registro');
    const progreso = $('registro-progreso');
    const barra = $('registro-barra');
    const mensaje = $('registro-mensaje');
    let temporizador = null;

    function terminar(recargar) {
      clearInterval(temporizador);
      temporizador = null;
      img.src = '';
      btn.disabled = false;
      btnCancelar.hidden = true;
      if (recargar) setTimeout(() => location.reload(), 1800);
    }

    btn.addEventListener('click', async () => {
      btn.disabled = true;
      try {
        await postJSON(btn.dataset.iniciar);
      } catch (e) { btn.disabled = false; alert('No se pudo iniciar el registro: ' + e.message); return; }
      marco.hidden = false; progreso.hidden = false; btnCancelar.hidden = false;
      img.src = btn.dataset.stream + '?t=' + Date.now();
      mensaje.textContent = 'Iniciando cámara…';

      temporizador = setInterval(async () => {
        try {
          const { registro: r } = await getJSON(btn.dataset.estado);
          if (!r) return;
          barra.style.width = r.progreso + '%';
          mensaje.textContent = `${r.guardadas}/${r.objetivo} muestras · ${r.mensaje}`;
          if (r.cancelada) terminar(false);
          else if (r.terminada && r.mensaje.startsWith('¡Listo!')) terminar(true);
          else if (r.terminada && r.mensaje.includes('falló')) terminar(false);
        } catch (_) { /* reintenta */ }
      }, 500);
    });

    btnCancelar.addEventListener('click', async () => {
      await postJSON(btn.dataset.cancelar).catch(() => {});
      mensaje.textContent = 'Registro cancelado.';
      terminar(false);
    });
  }

  return { paginaEnVivo, paginaRegistro };
})();
