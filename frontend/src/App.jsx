import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
    Send, Bot, User, FileText, Search,
    Loader2, Moon, Sun, Mic, MicOff,
    Maximize2, X, ExternalLink, ChevronDown, ChevronUp,
    Filter, FolderOpen, Trash2, WifiOff
} from 'lucide-react'
import './App.css'

const API = '/api'

// ── PDF Preview Modal ────────────────────────────────────────────────────
function PDFModal({ isOpen, fileUrl, fileName, onClose }) {
    if (!isOpen) return null
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-panel" onClick={e => e.stopPropagation()}>
                <div className="modal-bar">
                    <div className="modal-title">
                        <FileText size={15} />
                        <span>{fileName}</span>
                    </div>
                    <div className="modal-controls">
                        <a href={fileUrl} target="_blank" rel="noreferrer"
                            className="icon-btn" title="Open in new tab">
                            <ExternalLink size={15} />
                        </a>
                        <button className="icon-btn" onClick={onClose} title="Close">
                            <X size={16} />
                        </button>
                    </div>
                </div>
                <div className="modal-body">
                    <iframe src={fileUrl} title={fileName} />
                </div>
            </div>
        </div>
    )
}

// ── Top Bar ──────────────────────────────────────────────────────────────
function TopBar({ ready, dark, onToggleDark, onClearHistory, hasHistory }) {
    return (
        <header className="topbar">
            <div className="brand">
                <div className="brand-logo"><Search size={16} /></div>
                <div>
                    <div className="brand-name">Land Records Assistant</div>
                    <div className="brand-sub">Document Intelligence</div>
                </div>
            </div>
            <div className="topbar-right">
                {hasHistory && (
                    <button className="icon-btn" onClick={onClearHistory} title="Clear chat history">
                        <Trash2 size={15} />
                    </button>
                )}
                <button className="icon-btn" onClick={onToggleDark} title="Toggle theme">
                    {dark ? <Sun size={16} /> : <Moon size={16} />}
                </button>
                <div className={`status-pill ${ready ? 'online' : ''}`}>
                    <span className="status-dot" />
                    {ready ? 'Ready' : 'Loading'}
                </div>
            </div>
        </header>
    )
}

// ── Filter Bar ──────────────────────────────────────────────────────────
function FilterBar({ docs, activeFilter, onFilter }) {
    if (!docs || docs.length === 0) return null

    // Show just the filename (last path part); deduplicate by basename
    const getName = (path) => path.split(/[/\\]/).pop()
    const seen = new Set()
    const uniqueDocs = docs.filter(doc => {
        const name = getName(doc)
        if (seen.has(name)) return false
        seen.add(name)
        return true
    })

    return (
        <div className="filter-bar">
            <span className="filter-label">
                <Filter size={11} />
                Filter:
            </span>
            <button
                className={`filter-chip ${!activeFilter ? 'active' : ''}`}
                onClick={() => onFilter(null)}
            >
                All docs
            </button>
            {uniqueDocs.map((doc, i) => {
                const name = getName(doc)
                return (
                    <button
                        key={i}
                        className={`filter-chip ${activeFilter === doc ? 'active' : ''}`}
                        onClick={() => onFilter(activeFilter === doc ? null : doc)}
                        title={doc}
                    >
                        <FileText size={10} />
                        {name}
                    </button>
                )
            })}
        </div>
    )
}


// ── Discovery Card ───────────────────────────────────────────────────────
function DiscoveryCard({ table }) {
    if (!table?.length) return null

    const merged = {}
    table.forEach(row => {
        Object.entries(row).forEach(([k, v]) => {
            if (k !== 'Source' && !merged[k] && v && v !== '-') merged[k] = v
        })
    })

    const entries = Object.entries(merged)
    if (!entries.length) return null

    const sources = [...new Set(table.map(r => r.Source).filter(Boolean))]

    return (
        <div className="discovery-card">
            <div className="discovery-header">Extracted Information</div>
            <table className="discovery-table">
                <tbody>
                    {entries.map(([key, val]) => (
                        <tr key={key}>
                            <th>{key}</th>
                            <td>{val}</td>
                        </tr>
                    ))}
                    {sources.length > 0 && (
                        <tr>
                            <th>Source</th>
                            <td>{sources.join(', ')}</td>
                        </tr>
                    )}
                </tbody>
            </table>
        </div>
    )
}

// ── Read More Panel ──────────────────────────────────────────────────────
function ReadMorePanel({ detail }) {
    const [open, setOpen] = useState(false)
    if (!detail) return null

    return (
        <div className="read-more">
            <button
                className="read-more-toggle"
                onClick={() => setOpen(o => !o)}
            >
                {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                {open ? 'Show less' : 'Read more — detailed passage'}
            </button>
            {open && (
                <div className="read-more-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {detail}
                    </ReactMarkdown>
                </div>
            )}
        </div>
    )
}

// ── Message Bubble ───────────────────────────────────────────────────────
function MessageBubble({ msg, onPreview }) {
    const isBot = msg.role === 'bot'

    const getFileUrl = path => {
        const p = path.replace(/\\/g, '/')
        return p.split('/').length > 1
            ? `${API}/docs/samples/${p}`
            : `${API}/docs/uploads/${p}`
    }

    return (
        <div className={`message ${isBot ? 'bot' : 'user'}`}>
            <div className="avatar">
                {isBot ? <Bot size={15} /> : <User size={15} />}
            </div>

            <div className="bubble">
                {msg.loading ? (
                    <div className="skeleton">
                        <div className="skeleton-line" style={{ width: '85%' }} />
                        <div className="skeleton-line" style={{ width: '65%' }} />
                        <div className="skeleton-line" style={{ width: '45%' }} />
                    </div>
                ) : (
                    <>
                        {/* Structured extraction card */}
                        {isBot && <DiscoveryCard table={msg.table} />}

                        {/* Main answer */}
                        <div className="md-content">
                            {isBot ? (
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {msg.content || ''}
                                </ReactMarkdown>
                            ) : (
                                msg.content
                            )}
                        </div>

                        {/* Read More — expandable raw passage */}
                        {isBot && msg.detail && (
                            <ReadMorePanel detail={msg.detail} />
                        )}

                        {/* Source chips — deduplicated by filename */}
                        {isBot && (msg.sourcesDetail?.length > 0 || msg.sources?.length > 0) && (
                            <div className="sources">
                                <div className="sources-label">Sources</div>
                                {(() => {
                                    const raw = msg.sourcesDetail || msg.sources || []
                                    const seen = new Set()
                                    return raw
                                        .filter(s => {
                                            const path = typeof s === 'string' ? s : s.filename
                                            const name = path.split(/[/\\]/).pop()
                                            if (seen.has(name)) return false
                                            seen.add(name)
                                            return true
                                        })
                                        .map((s, i) => {
                                            const path = typeof s === 'string' ? s : s.filename
                                            const name = path.split(/[/\\]/).pop()
                                            return (
                                                <button
                                                    key={i}
                                                    className="source-chip"
                                                    onClick={() => onPreview(getFileUrl(path), name)}
                                                    title="View document"
                                                >
                                                    <FileText size={11} />
                                                    {name}
                                                    <Maximize2 size={10} />
                                                </button>
                                            )
                                        })
                                })()}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}

// ── Chat History Persistence ─────────────────────────────────────────────
const STORAGE_KEY = 'land-chat-history'
const MAX_STORED_MESSAGES = 50

const WELCOME_MSG = {
    id: 0,
    role: 'bot',
    content: `**Land Records Assistant**

कुछ भी पूछें — Hindi, Hinglish, या English में। भूखंड, जमाराशि, agency name, tender amount, work order — सब मिलेगा।

Filter bar से specific PDF चुन सकते हैं। Galat spelling भी चलेगी।`,
    sources: [],
    table: [],
    sourcesDetail: [],
    detail: null
}

function loadSavedMessages() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (raw) {
            const parsed = JSON.parse(raw)
            if (Array.isArray(parsed) && parsed.length > 0) return parsed
        }
    } catch { /* corrupted storage */ }
    return [WELCOME_MSG]
}

// ── App ──────────────────────────────────────────────────────────────────
export default function App() {
    const [messages, setMessages] = useState(loadSavedMessages)
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [ready, setReady] = useState(false)
    const [dark, setDark] = useState(localStorage.getItem('theme') === 'dark')
    const [listening, setListening] = useState(false)
    const [preview, setPreview] = useState({ open: false, url: '', name: '' })
    const [docList, setDocList] = useState([])
    const [activeFilter, setActiveFilter] = useState(null)
    const [connectionError, setConnectionError] = useState(false)

    const bottomRef = useRef(null)
    const inputRef = useRef(null)
    const recognRef = useRef(null)

    useEffect(() => { document.body.classList.toggle('dark', dark) }, [dark])

    // Persist messages to localStorage
    useEffect(() => {
        try {
            const toSave = messages.slice(-MAX_STORED_MESSAGES).map(m => ({
                ...m, loading: false // never persist loading state
            }))
            localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave))
        } catch { /* quota exceeded — fail silently */ }
    }, [messages])

    const checkStatus = useCallback(async () => {
        try {
            const r = await fetch(`${API}/status`)
            const d = await r.json()
            setReady(d.ready)
            setConnectionError(false)
            if (d.indexed_documents?.length) {
                setDocList(d.indexed_documents)
            }
        } catch {
            setConnectionError(true)
        }
    }, [])

    useEffect(() => { checkStatus() }, [checkStatus])

    // Auto-retry connection every 5 seconds when disconnected
    useEffect(() => {
        if (!connectionError) return
        const interval = setInterval(checkStatus, 5000)
        return () => clearInterval(interval)
    }, [connectionError, checkStatus])

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    // Voice recognition
    useEffect(() => {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition
        if (!SR) return
        const r = new SR()
        r.continuous = false
        r.interimResults = false
        r.lang = 'hi-IN'
        r.onresult = e => { setInput(e.results[0][0].transcript); setListening(false) }
        r.onerror = () => setListening(false)
        r.onend = () => setListening(false)
        recognRef.current = r
    }, [])

    const toggleListen = () => {
        if (listening) recognRef.current?.stop()
        else { setListening(true); recognRef.current?.start() }
    }

    const toggleDark = () => {
        const next = !dark
        setDark(next)
        localStorage.setItem('theme', next ? 'dark' : 'light')
    }

    const clearHistory = () => {
        setMessages([WELCOME_MSG])
        localStorage.removeItem(STORAGE_KEY)
    }

    const hasHistory = messages.length > 1
    const showSuggestions = messages.length <= 1

    const handleSend = async () => {
        const query = input.trim()
        if (!query || loading) return

        const userMsg = { id: Date.now(), role: 'user', content: query }
        const botId = Date.now() + 1
        const botMsg = {
            id: botId, role: 'bot', content: '',
            loading: true, table: [], sources: [], sourcesDetail: [], detail: null
        }

        setMessages(prev => [...prev, userMsg, botMsg])
        setInput('')
        // Reset textarea height back to single row after clearing
        if (inputRef.current) {
            inputRef.current.style.height = ''
        }
        setLoading(true)

        try {
            const res = await fetch(`${API}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    ...(activeFilter ? { filename_filter: activeFilter } : {}),
                })
            })
            if (!res.ok) throw new Error(`Server error ${res.status}`)

            const reader = res.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''

            setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, loading: false } : m
            ))

            while (true) {
                const { value, done } = await reader.read()
                if (done) break
                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n')
                buffer = lines.pop()

                for (const line of lines) {
                    if (!line.trim()) continue
                    try {
                        const chunk = JSON.parse(line)
                        setMessages(prev => prev.map(m => {
                            if (m.id !== botId) return m
                            if (chunk.type === 'meta')
                                return { ...m, table: chunk.table, sources: chunk.sources }
                            if (chunk.type === 'content')
                                return { ...m, content: m.content + chunk.content }
                            if (chunk.type === 'detail')
                                return { ...m, detail: chunk.content }
                            if (chunk.type === 'sources_detail')
                                return { ...m, sourcesDetail: chunk.content }
                            if (chunk.type === 'error')
                                return { ...m, content: chunk.content }
                            return m
                        }))
                    } catch { /* skip malformed line */ }
                }
            }
        } catch (e) {
            setMessages(prev => prev.map(m =>
                m.id === botId
                    ? { ...m, content: `Something went wrong: ${e.message}`, loading: false }
                    : m
            ))
        } finally {
            setLoading(false)
            setTimeout(() => inputRef.current?.focus(), 80)
        }
    }

    const onKey = e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
    }

    const autoGrow = e => {
        const el = e.target
        // Remove inline height first so scrollHeight is recalculated correctly
        el.style.height = ''
        if (el.value) {
            // Only set explicit height when there IS content
            el.style.height = Math.min(el.scrollHeight, 160) + 'px'
        }
        // When empty: leave style.height unset so CSS rows={1} controls it
    }

    const handleSuggestion = text => {
        setInput(text)
        setTimeout(() => inputRef.current?.focus(), 50)
    }

    const suggestions = [
        'भूखंड ka size kya hai?',
        'जमाराशि kitni hai?',
        'Agency name kya hai?',
        'Date of casting kya hai?',
        'Tender amount kitna tha?',
        'भूखंड 148 ki jankari do',
    ]

    const filterName = activeFilter ? activeFilter.split(/[/\\]/).pop() : null

    return (
        <div className="app">
            <TopBar
                ready={ready}
                dark={dark}
                onToggleDark={toggleDark}
                onClearHistory={clearHistory}
                hasHistory={hasHistory}
            />

            {/* Connection error toast */}
            {connectionError && (
                <div className="connection-toast">
                    <WifiOff size={14} />
                    Server unreachable — retrying...
                </div>
            )}

            {/* Document filter bar */}
            <FilterBar
                docs={docList}
                activeFilter={activeFilter}
                onFilter={setActiveFilter}
            />

            {/* Active filter badge */}
            {filterName && (
                <div className="active-filter-badge">
                    <FileText size={11} />
                    Filtering: <strong>{filterName}</strong>
                    <button onClick={() => setActiveFilter(null)} title="Clear filter">
                        <X size={12} />
                    </button>
                </div>
            )}

            <main className="chat-window">
                {messages.map(msg => (
                    <MessageBubble
                        key={msg.id}
                        msg={msg}
                        onPreview={(url, name) => setPreview({ open: true, url, name })}
                    />
                ))}
                {/* Suggestion chips — only on fresh chat */}
                {showSuggestions && (
                    <div className="suggestions">
                        {suggestions.map((text, i) => (
                            <button
                                key={i}
                                className="suggestion-chip"
                                onClick={() => handleSuggestion(text)}
                                disabled={loading}
                            >
                                {text}
                            </button>
                        ))}
                    </div>
                )}
                <div ref={bottomRef} />
            </main>

            <div className="input-area">
                <div className="input-box">
                    <textarea
                        ref={inputRef}
                        rows={1}
                        placeholder={listening
                            ? '🎤 Listening...'
                            : 'Ask anything… (Enter to send, Shift+Enter for new line)'}
                        value={input}
                        onChange={e => { setInput(e.target.value); autoGrow(e) }}
                        onKeyDown={onKey}
                        disabled={loading}
                    />
                    <div className="input-actions">
                        <button
                            className={`btn-icon ${listening ? 'listening' : ''}`}
                            onClick={toggleListen}
                            disabled={loading}
                            title="Voice input (Hindi)"
                        >
                            {listening ? <MicOff size={17} /> : <Mic size={17} />}
                        </button>
                        <button
                            className="btn-send"
                            onClick={handleSend}
                            disabled={loading || !input.trim()}
                            title="Send (Enter)"
                        >
                            {loading
                                ? <Loader2 size={17} className="animate-spin" />
                                : <Send size={17} />}
                        </button>
                    </div>
                </div>
                {input.length > 0 && (
                    <div className="input-hint">Shift+Enter for new line · Enter to send</div>
                )}
            </div>

            <PDFModal
                isOpen={preview.open}
                fileUrl={preview.url}
                fileName={preview.name}
                onClose={() => setPreview(p => ({ ...p, open: false }))}
            />
        </div>
    )
}
