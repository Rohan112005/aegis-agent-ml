
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Shield,
  Activity,
  Radar,
  Bell,
  Settings,
  RefreshCw,
  CircleAlert,
  CheckCircle2,
  Wifi,
  Cpu,
  Network,
  ArrowUpRight,
  LayoutDashboard,
  Clock3,
  Zap,
  Server,
  ChevronRight,
  X,
} from 'lucide-react';
import './style.css';

const API = import.meta.env.VITE_API_URL || '';

const DEFAULT_FEATURES =
  '38308, 1, 1, 6, 6, 313.250496, 52.208416, 0, 1, 9';

const DEFAULT_RAW = JSON.stringify(
  {
    source: 'dashboard test',
    protocol: 'TCP',
    destination_port: 443,
  },
  null,
  2
);

const NAV = [
  { name: 'Overview', icon: LayoutDashboard },
  { name: 'Live Traffic', icon: Radar },
  { name: 'Alerts', icon: Bell },
  { name: 'Agent Investigation', icon: Shield },
  { name: 'Settings', icon: Settings },
];

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail =
      typeof data.detail === 'string'
        ? data.detail
        : data.detail
          ? JSON.stringify(data.detail)
          : `Request failed with HTTP ${response.status}`;

    throw new Error(detail);
  }

  return data;
}

function Metric({ label, value, note, icon: Icon, tone = 'cyan' }) {
  return (
    <article className="metric card">
      <div className="metric-top">
        <span>{label}</span>
        <span className={`metric-icon ${tone}`}>
          <Icon size={18} />
        </span>
      </div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-note">{note}</span>
    </article>
  );
}

function EmptyState({ icon: Icon = Radar, title, description }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Icon size={25} />
      </div>
      <strong>{title}</strong>
      <p>{description}</p>
    </div>
  );
}

function App() {
  const [page, setPage] = useState('Overview');
  const [health, setHealth] = useState(null);
  const [dashboard, setDashboard] = useState(null);
  const [connection, setConnection] = useState('checking');
  const [lastChecked, setLastChecked] = useState(null);

  const [features, setFeatures] = useState(DEFAULT_FEATURES);
  const [raw, setRaw] = useState(DEFAULT_RAW);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');

  const refresh = useCallback(async () => {
    setConnection('checking');
    setError('');

    try {
      const [healthData, dashboardData] = await Promise.all([
        request('/health'),
        request('/api/dashboard'),
      ]);

      setHealth(healthData);
      setDashboard(dashboardData);
      setConnection('online');
    } catch (err) {
      setHealth(null);
      setDashboard(null);
      setConnection('offline');
      setError(
        err.message || 'Unable to connect to the AegisAgent backend.'
      );
    } finally {
      setLastChecked(new Date());
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const analyze = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    setResult(null);

    try {
      const values = features.split(',').map((value) => {
        const trimmed = value.trim();

        if (!trimmed) {
          throw new Error('Every feature must have a number.');
        }

        const number = Number(trimmed);

        if (!Number.isFinite(number)) {
          throw new Error('All features must be finite numbers.');
        }

        return number;
      });

      if (values.length !== 10) {
        throw new Error(
          `Enter exactly 10 numbers. You entered ${values.length}.`
        );
      }

      let rawLog;

      try {
        rawLog = JSON.parse(raw);
      } catch {
        throw new Error('The raw event context is not valid JSON.');
      }

      if (
        rawLog === null ||
        typeof rawLog !== 'object' ||
        Array.isArray(rawLog)
      ) {
        throw new Error('Raw event context must be a JSON object.');
      }

      const data = await request('/analyze-packet', {
        method: 'POST',
        body: JSON.stringify({
          telemetry_features: values,
          raw_log: rawLog,
        }),
      });

      if (typeof data.anomaly !== 'boolean') {
        throw new Error(
          'The API response did not contain a valid anomaly result.'
        );
      }

      const entry = {
        id: `${Date.now()}-${Math.random()}`,
        time: new Date().toISOString(),
        ...data,
      };

      setResult(data);
      setHistory((previous) => [entry, ...previous].slice(0, 100));
    } catch (err) {
      setError(err.message || 'Analysis request failed.');
    } finally {
      setBusy(false);
    }
  };

  const alerts = useMemo(
    () => history.filter((item) => item.anomaly === true),
    [history]
  );

  const filteredHistory = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) return history;

    return history.filter((item) =>
      [
        item.status,
        item.anomaly ? 'anomaly alert' : 'normal',
        item.score,
        item.model,
      ].some((value) =>
        String(value ?? '').toLowerCase().includes(query)
      )
    );
  }, [history, search]);

  const modelLoaded = health?.model_loaded === true;
  const threshold = health?.threshold;
  const captureStatus = dashboard?.capture_status || 'unknown';

  const resultScore =
    typeof result?.score === 'number' ? result.score : null;

  const resultThreshold =
    typeof result?.threshold === 'number'
      ? result.threshold
      : threshold;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Shield size={23} />
          </div>

          <div className="brand-copy">
            <b>
              AegisAgent<span>.</span>
            </b>
            <small>NETWORK DEFENSE</small>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="navigation">
          {NAV.map(({ name, icon: Icon }) => (
            <button
              key={name}
              className={`nav-item ${page === name ? 'active' : ''}`}
              onClick={() => setPage(name)}
            >
              <Icon size={18} />
              <span>{name}</span>

              {name === 'Alerts' && alerts.length > 0 && (
                <span className="nav-count">{alerts.length}</span>
              )}

              {page === name && (
                <ChevronRight className="nav-chevron" size={15} />
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-spacer" />

        <div className="model-mini">
          <div className="model-mini-icon">
            <Cpu size={18} />
          </div>

          <div>
            <strong>Detection engine</strong>
            <span>
              {modelLoaded
                ? 'Autoencoder loaded'
                : 'Model not confirmed'}
            </span>
          </div>

          <span
            className={`status-dot ${modelLoaded ? 'green' : 'amber'}`}
          />
        </div>

        <div className="sidebar-footer">
          <div className="connection-line">
            <span
              className={`status-dot ${
                connection === 'online'
                  ? 'green'
                  : connection === 'offline'
                    ? 'red'
                    : 'amber'
              }`}
            />
            <span>
              API{' '}
              {connection === 'online'
                ? 'Connected'
                : connection === 'offline'
                  ? 'Offline'
                  : 'Checking'}
            </span>
          </div>
          <small>AegisAgent · Local instance</small>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>AEGISAGENT</span>
            <span className="breadcrumb-separator">/</span>
            <span className="breadcrumb-current">
              {page.toUpperCase()}
            </span>
          </div>

          <div className="topbar-actions">
            <div className={`connection-badge ${connection}`}>
              <span className="status-dot" />
              {connection === 'online'
                ? 'Backend online'
                : connection === 'offline'
                  ? 'Backend offline'
                  : 'Checking backend'}
            </div>

            <button
              className="icon-button"
              onClick={refresh}
              disabled={connection === 'checking'}
              title="Refresh backend status"
              aria-label="Refresh backend status"
            >
              <RefreshCw
                size={17}
                className={connection === 'checking' ? 'spin' : ''}
              />
            </button>
          </div>
        </header>

        <section className="page-heading">
          <div>
            <div className="eyebrow">
              SECURITY OPERATIONS CENTER
            </div>
            <h1>
              {page === 'Overview' ? 'Security overview' : page}
            </h1>
            <p className="page-description">
              Monitor detection health and investigate network anomalies.
            </p>
          </div>

          <div className="heading-time">
            <Clock3 size={15} />
            <span>
              {lastChecked
                ? `Checked ${lastChecked.toLocaleTimeString()}`
                : 'Not checked yet'}
            </span>
          </div>
        </section>

        {connection === 'offline' && (
          <div className="notice notice-warning">
            <CircleAlert size={18} />
            <div>
              <strong>Backend is not reachable</strong>
              <p>
                Start the FastAPI server and check the configured port.
                Then refresh this dashboard.
              </p>
            </div>
            <button className="notice-action" onClick={refresh}>
              Retry
            </button>
          </div>
        )}

        {error && (
          <div className="error-message" role="alert">
            <CircleAlert size={18} />
            <span>{error}</span>
            <button
              onClick={() => setError('')}
              aria-label="Dismiss error"
            >
              <X size={16} />
            </button>
          </div>
        )}

        {page === 'Overview' && (
          <>
            <section className="metrics-grid">
              <Metric
                label="ENGINE STATUS"
                value={
                  connection === 'checking'
                    ? 'Checking'
                    : modelLoaded
                      ? 'Ready'
                      : 'Unavailable'
                }
                note={
                  modelLoaded
                    ? 'Autoencoder model loaded'
                    : 'Verify backend and model artifacts'
                }
                icon={Cpu}
                tone="cyan"
              />

              <Metric
                label="DETECTION THRESHOLD"
                value={
                  typeof threshold === 'number'
                    ? threshold.toExponential(3)
                    : '—'
                }
                note="Reported by the backend"
                icon={Activity}
                tone="purple"
              />

              <Metric
                label="CAPTURE STATUS"
                value={
                  captureStatus === 'not_started'
                    ? 'Stopped'
                    : captureStatus
                }
                note="Live capture is not started by this dashboard"
                icon={Wifi}
                tone="amber"
              />

              <Metric
                label="ANALYSES THIS SESSION"
                value={history.length}
                note="Requests made from this dashboard"
                icon={Network}
                tone="green"
              />
            </section>

            <section className="content-grid">
              <article className="card analysis-card">
                <div className="section-heading">
                  <div className="section-icon">
                    <Zap size={18} />
                  </div>
                  <div>
                    <h2>Traffic analysis sandbox</h2>
                    <p>
                      Submit a 10-feature vector to the existing
                      Autoencoder API.
                    </p>
                  </div>
                  <span className="section-tag">MANUAL TEST</span>
                </div>

                <form onSubmit={analyze}>
                  <label className="field-label" htmlFor="features">
                    Telemetry features
                    <span>10 numbers, comma-separated</span>
                  </label>

                  <textarea
                    id="features"
                    value={features}
                    onChange={(event) => setFeatures(event.target.value)}
                    rows={3}
                    spellCheck={false}
                    required
                  />

                  <label
                    className="field-label raw-label"
                    htmlFor="raw-context"
                  >
                    Raw event context
                    <span>Valid JSON object</span>
                  </label>

                  <textarea
                    id="raw-context"
                    value={raw}
                    onChange={(event) => setRaw(event.target.value)}
                    rows={4}
                    spellCheck={false}
                    required
                  />

                  <div className="form-footer">
                    <span className="helper-text">
                      Uses the existing /analyze-packet endpoint.
                    </span>
                    <button
                      className="primary-button"
                      type="submit"
                      disabled={busy}
                    >
                      {busy ? 'Analyzing…' : 'Analyze traffic'}
                      {busy ? (
                        <RefreshCw size={16} className="spin" />
                      ) : (
                        <ArrowUpRight size={16} />
                      )}
                    </button>
                  </div>
                </form>

                {result && (
                  <div
                    className={`analysis-result ${
                      result.anomaly ? 'danger' : 'safe'
                    }`}
                  >
                    <div className="result-header">
                      {result.anomaly ? (
                        <CircleAlert size={20} />
                      ) : (
                        <CheckCircle2 size={20} />
                      )}

                      <div>
                        <strong>
                          {result.anomaly
                            ? 'Anomaly detected'
                            : 'Traffic classified as normal'}
                        </strong>
                        <span>{result.status || 'Analysis completed'}</span>
                      </div>

                      <span className="result-score">
                        {resultScore === null
                          ? '—'
                          : resultScore.toExponential(3)}
                      </span>
                    </div>

                    <div className="score-track">
                      <div
                        className="score-fill"
                        style={{
                          width:
                            resultScore === null
                              ? '0%'
                              : `${Math.min(
                                  100,
                                  Math.max(0, resultScore * 100)
                                )}%`,
                        }}
                      />
                    </div>

                    <div className="result-meta">
                      <span>
                        Score{' '}
                        <b>
                          {resultScore === null
                            ? '—'
                            : resultScore.toExponential(3)}
                        </b>
                      </span>
                      <span>
                        Threshold{' '}
                        <b>
                          {typeof resultThreshold === 'number'
                            ? resultThreshold.toExponential(3)
                            : '—'}
                        </b>
                      </span>
                    </div>

                    {result.agent_response && (
                      <details className="agent-details">
                        <summary>
                          View agent investigation response
                        </summary>
                        <pre>
                          {JSON.stringify(result.agent_response, null, 2)}
                        </pre>
                      </details>
                    )}

                    <p className="disclaimer">
                      This is a model classification, not proof that
                      traffic is safe or malicious.
                    </p>
                  </div>
                )}
              </article>

              <div className="right-column">
                <article className="card system-card">
                  <div className="card-heading">
                    <h2>System health</h2>
                    <Server size={17} />
                  </div>

                  <div className="health-row">
                    <span className="health-indicator">
                      <span
                        className={`status-dot ${
                          connection === 'online' ? 'green' : 'red'
                        }`}
                      />
                      API service
                    </span>
                    <strong
                      className={
                        connection === 'online'
                          ? 'text-green'
                          : 'text-red'
                      }
                    >
                      {connection === 'online'
                        ? 'Online'
                        : connection === 'checking'
                          ? 'Checking'
                          : 'Offline'}
                    </strong>
                  </div>

                  <div className="health-row">
                    <span className="health-indicator">
                      <span
                        className={`status-dot ${
                          modelLoaded ? 'green' : 'amber'
                        }`}
                      />
                      Autoencoder
                    </span>
                    <strong>
                      {modelLoaded ? 'Loaded' : 'Unconfirmed'}
                    </strong>
                  </div>

                  <div className="health-row">
                    <span className="health-indicator">
                      <span
                        className={`status-dot ${
                          captureStatus === 'running' ? 'green' : 'amber'
                        }`}
                      />
                      Packet capture
                    </span>
                    <strong>
                      {captureStatus === 'running'
                        ? 'Running'
                        : 'Not started'}
                    </strong>
                  </div>

                  <button
                    className="secondary-button full-width"
                    onClick={refresh}
                    disabled={connection === 'checking'}
                  >
                    <RefreshCw size={15} />
                    Refresh health
                  </button>
                </article>

                <article className="card recent-card">
                  <div className="card-heading">
                    <h2>Recent analyses</h2>
                    <button
                      className="text-button"
                      onClick={() => setPage('Alerts')}
                    >
                      View alerts <ChevronRight size={14} />
                    </button>
                  </div>

                  {history.length === 0 ? (
                    <EmptyState
                      icon={Activity}
                      title="No analyses yet"
                      description="Run the traffic analysis sandbox to see results here."
                    />
                  ) : (
                    <div className="recent-list">
                      {history.slice(0, 4).map((item) => (
                        <div className="recent-item" key={item.id}>
                          <span
                            className={`event-icon ${
                              item.anomaly ? 'danger' : 'safe'
                            }`}
                          >
                            {item.anomaly ? (
                              <CircleAlert size={16} />
                            ) : (
                              <CheckCircle2 size={16} />
                            )}
                          </span>

                          <div className="recent-copy">
                            <strong>
                              {item.anomaly
                                ? 'Anomaly flagged'
                                : 'Normal classification'}
                            </strong>
                            <span>
                              {new Date(item.time).toLocaleTimeString()}
                            </span>
                          </div>

                          <span className="recent-score">
                            {typeof item.score === 'number'
                              ? item.score.toExponential(2)
                              : '—'}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </article>
              </div>
            </section>

            <section className="card activity-card">
              <div className="card-heading">
                <div>
                  <h2>Network activity</h2>
                  <p>
                    The current dashboard API does not provide a live
                    traffic stream.
                  </p>
                </div>
                <span className="section-tag">AWAITING SOURCE</span>
              </div>

              <EmptyState
                icon={Radar}
                title="No live flow stream connected"
                description="Use an authorized capture integration before expecting real-time flow records here."
              />
            </section>
          </>
        )}

        {page === 'Live Traffic' && (
          <section className="card page-card">
            <div className="section-heading">
              <div className="section-icon">
                <Radar size={18} />
              </div>
              <div>
                <h2>Live traffic monitor</h2>
                <p>
                  Live flow streaming is not currently provided by the
                  dashboard API.
                </p>
              </div>
            </div>

            <div className="live-status-box">
              <Radar size={34} />
              <h3>Capture not started</h3>
              <p>
                {dashboard?.message ||
                  'No capture status is available from the backend.'}
              </p>
              <p>
                This page does not start packet capture automatically.
              </p>
            </div>
          </section>
        )}

        {page === 'Alerts' && (
          <section className="card page-card">
            <div className="card-heading">
              <div>
                <h2>Detection alerts</h2>
                <p>
                  Alerts generated by analysis requests in this browser
                  session.
                </p>
              </div>
              <span className="section-tag">{alerts.length} ALERTS</span>
            </div>

            {alerts.length === 0 ? (
              <EmptyState
                icon={Bell}
                title="No anomaly alerts in this session"
                description="An alert appears here when an analysis reports an anomaly."
              />
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>TIME</th>
                      <th>STATUS</th>
                      <th>SCORE</th>
                      <th>THRESHOLD</th>
                      <th>MODEL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alerts.map((item) => (
                      <tr key={item.id}>
                        <td>{new Date(item.time).toLocaleString()}</td>
                        <td>
                          <span className="status-pill danger">
                            ANOMALY
                          </span>
                        </td>
                        <td>
                          {typeof item.score === 'number'
                            ? item.score.toExponential(4)
                            : '—'}
                        </td>
                        <td>
                          {typeof item.threshold === 'number'
                            ? item.threshold.toExponential(4)
                            : '—'}
                        </td>
                        <td>{item.model || 'Autoencoder API'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {page === 'Agent Investigation' && (
          <section className="card page-card">
            <div className="card-heading">
              <div>
                <h2>Agent investigation</h2>
                <p>
                  Investigation data returned by the backend after an
                  anomaly is detected.
                </p>
              </div>
            </div>

            {result?.agent_response ? (
              <pre className="json-view">
                {JSON.stringify(result.agent_response, null, 2)}
              </pre>
            ) : (
              <EmptyState
                icon={Shield}
                title="No investigation response available"
                description="Submit traffic in the sandbox. The backend invokes its agent graph when it detects an anomaly."
              />
            )}
          </section>
        )}

        {page === 'Settings' && (
          <section className="card page-card settings-page">
            <div className="card-heading">
              <div>
                <h2>Connection and model settings</h2>
                <p>
                  Read-only status from your current backend configuration.
                </p>
              </div>
            </div>

            <div className="settings-row">
              <div>
                <strong>API base URL</strong>
                <span>Frontend request destination</span>
              </div>
              <code>{API || 'Vite proxy (same origin)'}</code>
            </div>

            <div className="settings-row">
              <div>
                <strong>Health endpoint</strong>
                <span>Backend reachability and model state</span>
              </div>
              <code>/health</code>
            </div>

            <div className="settings-row">
              <div>
                <strong>Dashboard endpoint</strong>
                <span>Capture status and input dimensions</span>
              </div>
              <code>/api/dashboard</code>
            </div>

            <div className="settings-row">
              <div>
                <strong>Analysis endpoint</strong>
                <span>Existing Autoencoder analysis</span>
              </div>
              <code>/analyze-packet</code>
            </div>

            <div className="settings-row">
              <div>
                <strong>Model threshold</strong>
                <span>Reported by backend; not editable here</span>
              </div>
              <code>
                {typeof threshold === 'number' ? threshold : 'Unavailable'}
              </code>
            </div>

            <p className="settings-note">
              Model weights, thresholds, and training artifacts are not
              changed by this dashboard.
            </p>
          </section>
        )}

        <footer className="app-footer">
          <span>
            <Shield size={14} /> AEGISAGENT · SECURITY CONSOLE
          </span>
          <span>LOCAL INSTANCE · MODEL OUTPUTS REQUIRE REVIEW</span>
        </footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
