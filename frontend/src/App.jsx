import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
    Send,
    Bot,
    User,
    FileText,
    Search,
    Loader2,
    Moon,
    Sun,
    Mic,
    MicOff,
    Maximize2,
    X,
    ExternalLink
} from 'lucide-react'
import './App.css'

const API = '/api'

// --- Components ---

function PDFModal({ isOpen, fileUrl, fileName, onClose }) {
    if (!isOpen) return null

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <div className="modal-title">
                        <FileText size={18} />
                        <span>{fileName}</span>
                    </div>
                    <div className="modal-actions">
                        <a href={fileUrl} target="_blank" rel="noreferrer" className="action-btn">
                            <ExternalLink size={18} />
                        </a>
                        <button onClick={onClose} className="action-btn close">
                            <X size={20} />
                        </button>
                    </div>
                </div>
                <div className="modal-body">
                    <iframe src={fileUrl} title={fileName} width="100%" height="100%" />
                </div>
            </div>
        </div>
    )
}

function Header({ ready, darkMode, onToggleDark }) {
    return (
        <header>
            <div className="brand">
                <div className="brand-icon">
                    <Search size={20} />
                </div>
                <h1>Land Records Assistant</h1>
            </div>
            <div className="header-right">
                <button className="icon-btn theme-toggle" onClick={onToggleDark} title="Toggle Dark Mode">
                    {darkMode ? <Sun size={20} /> : <Moon size={20} />}
                </button>
                <div className={`status-badge ${ready ? 'ready' : ''}`}>
                    <div className="status-dot" />
                    {ready ? 'System Ready' : 'Initializing...'}
                </div>
            </div>
        </header>
    )
}

function MessageBubble({ msg, onPreview }) {
    const isBot = msg.role === 'bot'

    const getFileUrl = (path) => {
        // Ensure we handle both backslashes and forward slashes
        const normalizedPath = path.replace(/\\/g, '/')

        // If it's a local sample PDF (has category/case structure), it's in /docs/samples
        if (normalizedPath.split('/').length > 1) {
            return `${API}/docs/samples/${normalizedPath}`
        }
        // Fallback for simple uploads
        return `${API}/docs/uploads/${normalizedPath}`
    }

    return (
        <div className={`message ${isBot ? 'bot' : 'user'}`}>
            <div className="avatar">
                {isBot ? <Bot size={18} /> : <User size={18} />}
            </div>
            <div className="bubble">
                {msg.loading ? (
                    <div className="typing-dots">
                        <span /><span /><span />
                    </div>
                ) : (
                    <div className="bubble-text">
                        {/* 1. Summary Table (if exists) */}
                        {isBot && msg.table?.length > 0 && (
                            <div className="summary-table-container">
                                <h3>📊 Key Discovery</h3>
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Source</th>
                                            {/* Extract all unique keys from all rows, excluding 'Source' */}
                                            {Array.from(new Set(msg.table.flatMap(row => Object.keys(row)).filter(k => k !== 'Source')))
                                                .map(key => <th key={key}>{key}</th>)}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {msg.table.map((row, i) => {
                                            const otherKeys = Array.from(new Set(msg.table.flatMap(r => Object.keys(r)).filter(k => k !== 'Source')));
                                            return (
                                                <tr key={i}>
                                                    <td>{row.Source}</td>
                                                    {otherKeys.map(key => <td key={key}>{row[key] || "-"}</td>)}
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        )}

                        {/* 2. Main AI Content */}
                        <div className="main-content">
                            {isBot ? (
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {msg.content || (msg.loading ? '...' : '')}
                                </ReactMarkdown>
                            ) : (
                                msg.content
                            )}
                        </div>

                        {/* 3. Source citations with previews */}
                        {isBot && (msg.sources?.length > 0 || msg.sourcesDetail?.length > 0) && (
                            <div className="sources">
                                <div className="sources-label">Sources:</div>
                                {(msg.sourcesDetail || msg.sources || []).map((s, i) => {
                                    const path = typeof s === 'string' ? s : s.filename
                                    const name = path.split(/[\\/]/).pop()
                                    const url = getFileUrl(path)
                                    return (
                                        <button
                                            key={i}
                                            className="source-tag clickable"
                                            onClick={() => onPreview(url, name)}
                                            title="Open Preview"
                                        >
                                            <FileText size={12} />
                                            {name}
                                            <Maximize2 size={10} style={{ marginLeft: '4px' }} />
                                        </button>
                                    )
                                })}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}

export default function App() {
    const [messages, setMessages] = useState([
        {
            id: 0,
            role: 'bot',
            content: '## Namaste! 🙏\n\nMain aapka Land Records Assistant hoon. Main aapke PDFs se **100% accurate extraction** aur **Key Discovery tables** generate karta hoon.\n\nAap mic button ka use karke bhi sawaal pooch sakte hain!',
            sources: []
        }
    ])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [ready, setReady] = useState(false)
    const [darkMode, setDarkMode] = useState(localStorage.getItem('theme') === 'dark')
    const [isListening, setIsListening] = useState(false)

    // PDF Preview State
    const [previewFile, setPreviewFile] = useState({ isOpen: false, url: '', name: '' })

    const bottomRef = useRef(null)
    const inputRef = useRef(null)
    const recognition = useRef(null)

    const checkStatus = useCallback(async () => {
        try {
            const res = await fetch(`${API}/status`)
            const data = await res.json()
            setReady(data.ready)
        } catch (err) {
            console.error("Status check failed:", err)
        }
    }, [])

    useEffect(() => {
        checkStatus()
        if (darkMode) {
            document.body.classList.add('dark-mode')
        } else {
            document.body.classList.remove('dark-mode')
        }
    }, [checkStatus, darkMode])

    useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

    // --- Voice Recognition ---
    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
        if (SpeechRecognition) {
            recognition.current = new SpeechRecognition()
            recognition.current.continuous = false
            recognition.current.interimResults = false
            recognition.current.lang = 'hi-IN' // Support Hindi/English

            recognition.current.onresult = (event) => {
                const transcript = event.results[0][0].transcript
                setInput(transcript)
                setIsListening(false)
            }

            recognition.current.onerror = (event) => {
                console.error("Speech Recognition Error:", event.error)
                setIsListening(false)
            }

            recognition.current.onend = () => {
                setIsListening(false)
            }
        }
    }, [])

    const toggleListen = () => {
        if (isListening) {
            recognition.current?.stop()
        } else {
            setIsListening(true)
            recognition.current?.start()
        }
    }

    const toggleDarkMode = () => {
        const newMode = !darkMode
        setDarkMode(newMode)
        localStorage.setItem('theme', newMode ? 'dark' : 'light')
    }

    const handleSend = async () => {
        const query = input.trim()
        if (!query || loading) return

        const userMsg = { id: Date.now(), role: 'user', content: query }
        const botId = Date.now() + 1
        const botMsg = {
            id: botId,
            role: 'bot',
            content: '',
            sources: [],
            loading: true,
            table: [],
            sourcesDetail: []
        }

        setMessages(prev => [...prev, userMsg, botMsg])
        setInput('')
        setLoading(true)

        try {
            const response = await fetch(`${API}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            })

            if (!response.ok) throw new Error('Server error')

            const reader = response.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''

            setMessages(prev => prev.map(m => m.id === botId ? { ...m, loading: false } : m))

            while (true) {
                const { value, done } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n')
                buffer = lines.pop() // keep partial line in buffer

                for (const line of lines) {
                    if (!line.trim()) continue
                    try {
                        const chunk = JSON.parse(line)

                        setMessages(prev => prev.map(m => {
                            if (m.id !== botId) return m

                            if (chunk.type === 'meta') {
                                return { ...m, table: chunk.table, sources: chunk.sources }
                            }
                            if (chunk.type === 'content') {
                                return { ...m, content: m.content + chunk.content }
                            }
                            if (chunk.type === 'sources_detail') {
                                return { ...m, sourcesDetail: chunk.content }
                            }
                            if (chunk.type === 'error') {
                                return { ...m, content: `### ❌ Error\n\n${chunk.content}` }
                            }
                            return m
                        }))
                    } catch (e) {
                        console.error("Error parsing stream chunk:", e)
                    }
                }
            }
        } catch (e) {
            setMessages(prev => prev.map(m =>
                m.id === botId ? {
                    ...m,
                    content: `### ❌ Error\n\n${e.message}`,
                    loading: false
                } : m
            ))
        } finally {
            setLoading(false)
            setTimeout(() => inputRef.current?.focus(), 100)
        }
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="app-container">
            <Header ready={ready} darkMode={darkMode} onToggleDark={toggleDarkMode} />

            <main className="chat-window">
                {messages.map(msg => (
                    <MessageBubble
                        key={msg.id}
                        msg={msg}
                        onPreview={(url, name) => setPreviewFile({ isOpen: true, url, name })}
                    />
                ))}
                <div ref={bottomRef} />
            </main>

            <div className="input-area">
                <div className="input-container">
                    <textarea
                        ref={inputRef}
                        placeholder={isListening ? "Listening..." : "Search land records or ask a question..."}
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        rows={1}
                        disabled={loading}
                    />
                    <div className="input-actions">
                        <button
                            className={`action-btn mic-btn ${isListening ? 'active' : ''}`}
                            onClick={toggleListen}
                            disabled={loading}
                            title="Voice Search"
                        >
                            {isListening ? <MicOff size={20} /> : <Mic size={20} />}
                        </button>
                        <button
                            className="send-btn"
                            onClick={handleSend}
                            disabled={loading || !input.trim()}
                        >
                            {loading ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
                        </button>
                    </div>
                </div>
            </div>

            <PDFModal
                isOpen={previewFile.isOpen}
                fileUrl={previewFile.url}
                fileName={previewFile.name}
                onClose={() => setPreviewFile({ ...previewFile, isOpen: false })}
            />
        </div>
    )
}
