'use strict';

const $ = id => document.getElementById(id);
const map = L.map('map', {zoomControl: false, preferCanvas: true, minZoom: 2, maxZoom: 16,
  maxBounds: [[22, -180], [85, -48]], maxBoundsViscosity: 0.8}).setView([53.02, -117.31], 10);
L.control.zoom({position: 'topright'}).addTo(map);
const tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);
const layers = {active: L.layerGroup().addTo(map), burned: L.layerGroup().addTo(map), candidate: L.layerGroup().addTo(map)};
let history = [], cursor = 0, ignitions = [], selectedCell = null, placing = false;
let playing = false, busy = false, timer = null, generation = 0, controller = null, config = null;
let source = 'placed';
let loadingFirms = false;
const HISTORY_LIMIT = 128;
const current = () => history[cursor];

function appendFrame(frame) {
  history.push(frame);
  if (history.length > HISTORY_LIMIT) history.shift();
  cursor = history.length - 1;
}

function message(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
  $('status').hidden = !text;
}

async function api(path, body, signal) {
  const response = await fetch(path, {method: body ? 'POST' : 'GET',
    headers: body ? {'Content-Type': 'application/json'} : {},
    body: body ? JSON.stringify(body) : undefined, signal});
  let data;
  try { data = await response.json(); } catch { throw new Error('The server returned an unreadable response. Try again.'); }
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map(e => e.msg).join('; ') : data.detail;
    throw new Error(detail || 'The request failed. Try again.');
  }
  return data;
}

// Requests are stateless. Pause invalidates and aborts an in-flight step so
// its eventual server response cannot move the visible timeline after pause.
function pause() {
  playing = false;
  clearTimeout(timer);
  generation++;
  if (controller) controller.abort();
  controller = null;
  busy = false;
  loadingFirms = false;
  updateControls();
}

function setPlacing(value) {
  placing = value;
  $('place').classList.toggle('placing', value);
  $('place').textContent = value ? '× Finish placing' : '+ Place on map';
  $('map').classList.toggle('placing-map', value);
  $('map-instruction').textContent = value ? 'Click the map to add fire cells. Click again to build a starting front.' :
    history.length ? 'Click a fire cell to pause and inspect.' : 'Start a scenario with a fire or current satellite detections.';
}

function updateControls() {
  const frame = current(), ready = config?.model_ready;
  const canAdvance = !!frame;
  $('play').disabled = !ready || !canAdvance;
  $('play').innerHTML = playing || busy ? 'Ⅱ <span>Pause</span>' : '▶ <span>Play</span>';
  $('step').disabled = !ready || !canAdvance || busy || playing;
  $('reset').disabled = !history.length && !busy;
  $('timeline').disabled = history.length < 2;
  const first = history[0]?.state.step_index ?? 0, last = history.at(-1)?.state.step_index ?? 0;
  $('timeline').min = first;
  $('timeline').max = Math.max(first + 1, last);
  $('timeline').value = frame?.state.step_index ?? 0;
  $('timeline-heading').textContent = first ? `RECENT TIMELINE · ${history.length} STEPS` : 'SIMULATION TIMELINE';
  document.querySelectorAll('.timeline-labels span').forEach((label, i) => {
    const step = Math.round(first + (last - first) * i / 4);
    label.textContent = (last - first < 4 && i > 0 && i < 4) || (!last && i > 0) ? '' : step ? `+${step * 12} h` : 'Start';
  });
  $('place').disabled = !ready || busy || playing || (!!frame && (frame.state.step_index > 0 || source !== 'placed'));
  $('coordinate-place').disabled = $('place').disabled;
  $('placement-hint').textContent = frame && (frame.state.step_index > 0 || source !== 'placed') ?
    'Reset the scenario to place new fires. You can still inspect completed steps on the timeline.' :
    'Choose an intensity, then click the map to add a starting fire.';
  $('load-firms').disabled = !ready || busy || playing || !config?.firms_configured;
  $('load-firms').textContent = loadingFirms ? 'Loading satellites…' : '↓ Load current FIRMS';
  $('state-badge').textContent = playing ? 'RUNNING' : frame ? 'PAUSED' : 'READY';
  $('playback-state').textContent = busy ? 'PREDICTING' : playing ? 'PLAYING' : 'PAUSED';
}

function draw() {
  Object.values(layers).forEach(layer => layer.clearLayers());
  const frame = current();
  $('active-count').textContent = frame?.active_count ?? 0;
  $('burned-count').textContent = frame?.burned_count ?? 0;
  $('new-count').textContent = frame?.new_ignition_count ?? '—';
  $('elapsed').textContent = `+${frame?.elapsed_hours ?? 0} hours`;
  $('valid-time').textContent = frame ? `${new Date(frame.valid_at).toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC'})} UTC · +${frame.elapsed_hours} h` : 'Place a fire to begin';
  if (frame) {
    const view = map.getBounds().pad(.1);
    const visible = frame.points.filter(p => view.contains([p.latitude, p.longitude]));
    const groups = new Map();
    for (const point of visible) {
      const shown = $(point.status === 'candidate' ? 'show-candidates' : `show-${point.status}`).checked;
      if (!shown) continue;
      const pixel = map.project([point.latitude, point.longitude]);
      const key = visible.length > 1500 && map.getZoom() < 11 && point.cell_id !== selectedCell ?
        `${point.status}:${Math.floor(pixel.x / 32)}:${Math.floor(pixel.y / 32)}` : point.cell_id;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(point);
    }
    for (const group of groups.values()) {
      const point = group[0];
      if (group.length > 1) {
        const lat = group.reduce((sum, p) => sum + p.latitude, 0) / group.length;
        const lon = group.reduce((sum, p) => sum + p.longitude, 0) / group.length;
        L.circleMarker([lat, lon], {radius: Math.min(20, 7 + Math.log2(group.length)),
          color: point.status === 'active' ? '#b83e21' : '#66634f', weight: 1,
          fillColor: point.status === 'active' ? '#e37040' : '#a69a7e', fillOpacity: .8, bubblingMouseEvents: false})
          .bindTooltip(`${group.length.toLocaleString()} ${point.status} cells · click to expand`)
          .on('click', () => {
            pause(); setPlacing(false);
            map.fitBounds(group.map(p => [p.latitude, p.longitude]), {padding: [70, 70], maxZoom: 12});
          }).addTo(layers[point.status]);
        continue;
      }
      const color = point.status === 'active' ? '#b83e21' : point.status === 'burned' ? '#5d5a52' : '#a8864f';
      const fillColor = point.status === 'active' ? `hsl(${12 + (1 - point.intensity) * 24}, 85%, ${48 + (1 - point.intensity) * 15}%)` : point.status === 'burned' ? '#767169' : '#efd5a6';
      const marker = L.circleMarker([point.latitude, point.longitude], {radius: point.status === 'active' ? 5 + point.intensity * 4 : point.status === 'burned' ? 5 : 4,
        color, weight: selectedCell === point.cell_id ? 3 : 1, fillColor,
        fillOpacity: point.status === 'candidate' ? .45 : .88, bubblingMouseEvents: false});
      marker.bindTooltip(`${point.status === 'candidate' ? 'Spread candidate' : point.status === 'active' ? 'Active fire' : 'Burned cell'} · click to inspect`);
      marker.on('click', () => { pause(); setPlacing(false); selectedCell = point.cell_id; inspect(point); draw(); });
      marker.addTo(layers[point.status]);
    }
    if (selectedCell) {
      const selected = frame.points.find(p => p.cell_id === selectedCell);
      if (selected) inspect(selected); else $('inspector').hidden = true;
    }
  } else $('inspector').hidden = true;
  updateControls();
}

function inspect(point) {
  $('inspector').hidden = false;
  $('point-title').textContent = point.status === 'active' ? point.new_ignition ? 'New ignition' : 'Active fire' : point.status === 'burned' ? 'Burned cell' : 'Spread candidate';
  $('point-location').textContent = `${point.latitude.toFixed(5)}°, ${point.longitude.toFixed(5)}°`;
  $('point-id').textContent = point.cell_id;
  const rows = [['Source', point.source], ['Simulated time', `+${current().elapsed_hours} hours`], ['Cell area', '1 km²']];
  if (point.intensity !== null) rows.push(['Scenario intensity', `${Math.round(point.intensity * 100)}%`]);
  if (point.ignition_probability !== null) rows.push(['Last-step spread probability', `${(point.ignition_probability * 100).toFixed(1)}%`]);
  if (point.observation_age_hours !== null) rows.push(['Observation age', `${point.observation_age_hours.toFixed(1)} h`]);
  if (point.detection_count !== null) rows.push(['FIRMS detections', point.detection_count]);
  if (point.bright_ti4_max !== null) rows.push(['Maximum brightness', `${point.bright_ti4_max.toFixed(1)} K`]);
  $('point-details').replaceChildren(...rows.map(([key, value]) => {
    const row = document.createElement('div'), term = document.createElement('dt'), definition = document.createElement('dd');
    term.textContent = key; definition.textContent = value; row.append(term, definition); return row;
  }));
}

async function addIgnition(latitude, longitude) {
  if ($('place').disabled) return;
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return message('Enter a valid latitude and longitude.', true);
  if (ignitions.length >= 500) return message('This preview supports at most 500 starting points.', true);
  pause(); busy = true; updateControls();
  const token = generation, next = [...ignitions, {latitude, longitude, intensity: Number($('intensity').value) / 100}];
  controller = new AbortController();
  try {
    const result = await api('/api/seed', {ignitions: next}, controller.signal);
    if (token !== generation) return;
    ignitions = next; source = 'placed'; history = [result]; cursor = 0;
    message(''); draw();
    $('map-instruction').textContent = 'Starting fire added. Add more cells, or press Play to predict spread.';
  } catch (e) { if (e.name !== 'AbortError' && token === generation) message(e.message, true); }
  finally { if (token === generation) { busy = false; controller = null; updateControls(); } }
}

async function advance() {
  if (busy || !current()) return;
  setPlacing(false);
  const token = generation;
  if (cursor < history.length - 1) {
    cursor++;
    draw(); return;
  }
  busy = true; updateControls(); controller = new AbortController();
  const frame = current();
  try {
    const result = await api('/api/step', {state: frame.state, origin_at: frame.origin_at}, controller.signal);
    if (token !== generation) return;
    appendFrame(result);
    message(result.extinct ? 'No active fire remains. The clock can continue; burned cells stay masked.' :
      result.terrain_missing_count ? `${result.terrain_missing_count} candidate cells have no retained terrain coverage; their terrain inputs are missing.` : '');
    draw();
  } catch (e) {
    if (e.name !== 'AbortError' && token === generation) { playing = false; message(e.message, true); }
  } finally {
    if (token === generation) { busy = false; controller = null; updateControls(); }
  }
}

function schedule() {
  clearTimeout(timer);
  if (!playing) return;
  timer = setTimeout(async () => {
    const started = generation;
    await advance();
    if (started === generation && playing) schedule();
  }, Number($('speed').value) * 1000);
}

$('play').addEventListener('click', () => {
  if (playing || busy) { pause(); return; }
  setPlacing(false); playing = true; message(''); updateControls(); schedule();
});
$('step').addEventListener('click', () => { pause(); advance(); });
$('speed').addEventListener('change', schedule);
$('timeline').addEventListener('input', () => {
  const requested = Number($('timeline').value);
  pause(); cursor = Math.max(0, Math.min(requested - history[0].state.step_index, history.length - 1)); draw();
  message('');
});
$('reset').addEventListener('click', () => {
  pause(); history = []; cursor = 0; ignitions = []; selectedCell = null; source = 'placed';
  setPlacing(false); message(''); draw();
});
$('place').addEventListener('click', () => { pause(); setPlacing(!placing); });
$('coordinate-place').addEventListener('click', () => {
  const lat = Number($('latitude').value), lon = Number($('longitude').value);
  addIgnition(lat, lon).then(() => { if (lat >= 24 && lat <= 84 && lon >= -179 && lon <= -50) map.setView([lat, lon], 11); });
});
map.on('click', e => { if (placing) addIgnition(e.latlng.lat, e.latlng.lng); });
map.on('moveend', () => {
  const center = map.getCenter();
  $('map-title').textContent = 'NORTH AMERICA';
  $('map-coordinates').textContent = `${center.lat.toFixed(3)}° N · ${Math.abs(center.lng).toFixed(3)}° W`;
  draw();
});
$('intensity').addEventListener('input', () => { $('intensity-value').value = `${$('intensity').value}%`; });
for (const id of ['show-active', 'show-burned', 'show-candidates']) $(id).addEventListener('change', draw);
$('close-inspector').addEventListener('click', () => { selectedCell = null; $('inspector').hidden = true; draw(); });
$('fit').addEventListener('click', () => {
  const points = current()?.points.filter(p => p.status !== 'candidate') || [];
  if (points.length) map.fitBounds(points.map(p => [p.latitude, p.longitude]), {padding: [85, 85], maxZoom: 12});
  else message('Place a fire or load FIRMS detections first.');
});
for (const mode of ['place', 'firms']) $(`${mode}-tab`).addEventListener('click', () => {
  pause(); setPlacing(false);
  for (const name of ['place', 'firms']) {
    $(`${name}-tab`).classList.toggle('selected', name === mode);
    $(`${name}-tab`).setAttribute('aria-pressed', name === mode);
    $(`${name}-panel`).hidden = name !== mode;
  }
  if (mode === 'firms' && config && !config.firms_configured) message('FIRMS requires NASA_FIRMS_API_KEY or MAP_KEY in the server’s config/.env. Fire placement is available.');
});
$('load-firms').addEventListener('click', async () => {
  pause(); setPlacing(false); busy = true; loadingFirms = true; updateControls();
  const bounds = map.getBounds(), token = generation;
  const [west, south, east, north] = config.firms_bounds;
  const requestBounds = $('firms-scope').value === 'all' ? {west, south, east, north} :
    {west: Math.max(west, bounds.getWest()), south: Math.max(south, bounds.getSouth()),
      east: Math.min(east, bounds.getEast()), north: Math.min(north, bounds.getNorth())};
  controller = new AbortController(); message('Loading current observations from three VIIRS satellites…');
  try {
    const result = await api('/api/firms', requestBounds, controller.signal);
    if (token !== generation) return;
    history = [result]; cursor = 0; ignitions = []; selectedCell = null; source = 'firms';
    const meta = result.metadata;
    message(`${result.active_count ? `Loaded ${meta.eligible_detection_count} observations in ${result.active_count} active cells.` : 'No eligible detections in this view.'} ${meta.recent_detections_excluded} observations newer than 3 hours excluded. Snapshot: ${new Date(meta.as_of).toLocaleString()}.`);
    setPlacing(false); draw();
    if (result.active_count) $('fit').click();
  } catch (e) { if (e.name !== 'AbortError' && token === generation) message(e.message, true); }
  finally { if (token === generation) { busy = false; loadingFirms = false; controller = null; updateControls(); } }
});
$('help').addEventListener('click', () => { pause(); $('help-dialog').showModal(); });
$('close-help').addEventListener('click', () => $('help-dialog').close());
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { pause(); setPlacing(false); }
  if (e.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement.tagName) && !$('help-dialog').open) {
    e.preventDefault(); if (!$('play').disabled) $('play').click();
  }
});
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
let tileErrorShown = false;
tiles.on('tileerror', () => { if (!tileErrorShown) { tileErrorShown = true; message('Map tiles are unavailable. Check your connection; placement by coordinates and simulation still work.', true); } });
api('/api/config').then(data => {
  config = data; $('model-name').textContent = data.model_name;
  if (!data.model_ready) message(data.model_error, true);
  updateControls();
}).catch(e => message(`Could not connect to the model server. ${e.message}`, true));
