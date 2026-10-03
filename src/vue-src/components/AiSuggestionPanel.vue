<script setup lang="ts">
import { ref, computed, onUnmounted } from 'vue'
import { api } from '../utils/api'

const emit = defineEmits<{
  applied: []
}>()

interface Suggestion {
  id: number
  name: string
  ocr: string
  filename: string
}

const visible = ref(false)
const running = ref(false)
const pct = ref(0)
const total = ref(0)
const message = ref('准备中...')
const errorMsg = ref('')
const taskId = ref('')
const rows = ref<Suggestion[]>([])
const selected = ref<Set<number>>(new Set())
// 本次标注的目标张数（0 表示按未标注增量），用于「仅标注选中的 N 张」提示
const scopeCount = ref(0)

let pollTimer: ReturnType<typeof setInterval> | null = null

function showToast(msg: string) {
  const el = document.getElementById('toast')
  if (!el) return
  el.textContent = msg
  el.classList.add('show')
  setTimeout(() => el.classList.remove('show'), 1600)
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

const allChecked = computed(
  () => rows.value.length > 0 && rows.value.every(r => selected.value.has(r.id)),
)

function thumbUrl(row: Suggestion): string {
  return `/api/thumb/${row.id}/${encodeURIComponent(row.filename)}`
}

// 打开面板：选中项优先标注；无选中时若已有待确认建议则直接展示（不重复消耗额度）
async function open(memeIds?: number[]) {
  visible.value = true
  errorMsg.value = ''
  rows.value = []
  selected.value = new Set()
  pct.value = 0
  total.value = 0
  scopeCount.value = memeIds && memeIds.length ? memeIds.length : 0
  stopPolling()

  const r = await startNew(memeIds)
  if (r) return

  // 无法启动（多半是未配置 AI 服务）：退回展示已有建议，避免面板空白
  const pending = await api('get_ai_suggestions', null)
  if (pending && Object.keys(pending).length) {
    const st = await api('ai_get_progress')
    taskId.value = (st && st.task_id) || ''
    setRows(pending)
    running.value = false
    errorMsg.value = ''
    message.value = `有 ${rows.value.length} 条待确认的建议`
    return
  }
  if (!errorMsg.value) errorMsg.value = '启动失败'
}

// 启动新任务；返回是否成功（失败时 errorMsg 已填好）
async function startNew(memeIds?: number[]): Promise<boolean> {
  running.value = true
  errorMsg.value = ''
  message.value = '准备中...'
  // 每次启动都只认当前传入的选中项，绝不复用上一次的目标集合
  const ids = memeIds && memeIds.length ? memeIds : null
  scopeCount.value = ids ? ids.length : 0
  message.value = ids ? `准备标注选中的 ${ids.length} 张...` : '准备中...'
  const r = await api('ai_organize', null, ids)
  if (!r || !r.ok) {
    running.value = false
    errorMsg.value = (r && r.error) || '启动失败'
    return false
  }
  // 后端已有任务在跑：不当作新任务，改为接入那个任务的进度
  if (r.started === false) {
    taskId.value = r.task_id || ''
    stopPolling()
    pollTimer = setInterval(poll, 400)
    showToast('已有标注任务正在进行，已接入其进度')
    return true
  }
  taskId.value = r.task_id || ''
  stopPolling()
  pollTimer = setInterval(poll, 400)
  return true
}

function setRows(data: Record<string, any>) {
  const list: Suggestion[] = []
  for (const v of Object.values(data || {})) {
    list.push({
      id: Number(v.id),
      name: String(v.name || ''),
      ocr: String(v.ocr || ''),
      filename: String(v.filename || ''),
    })
  }
  rows.value = list
  selected.value = new Set(list.map(r => r.id))
}

async function poll() {
  const s = await api('ai_get_progress')
  if (!s) return
  // 双保险：后端已保证单调，前端再取一次 max，避免旧响应晚到造成回跳
  pct.value = Math.max(pct.value, Number(s.progress) || 0)
  total.value = Number(s.total) || 0
  message.value = s.message || ''
  if (s.task_id) taskId.value = s.task_id
  if (!s.status || s.status === 'idle' || s.status === 'running') return
  stopPolling()
  running.value = false
  pct.value = 100
  if (s.status === 'error') {
    errorMsg.value = s.error || s.message || '标注失败'
    // 熔断中止时已成功的建议仍可用，先加载出来供审阅
    await loadSuggestions()
    if (rows.value.length) showToast('已中止，已完成的部分可继续审阅')
    return
  }
  await loadSuggestions()
  showToast(s.status === 'cancelled' ? '已取消，保留已完成部分' : '建议已生成')
}

function cancelTask() {
  api('ai_cancel', taskId.value || null)
}

async function loadSuggestions() {
  const data = await api('get_ai_suggestions', taskId.value || null)
  setRows(data || {})
}

function toggleRow(id: number, checked: boolean) {
  const next = new Set(selected.value)
  if (checked) next.add(id)
  else next.delete(id)
  selected.value = next
}

function toggleAll(checked: boolean) {
  selected.value = checked ? new Set(rows.value.map(r => r.id)) : new Set()
}

// 输入框失焦即同步到后端建议集（保证「应用」用的是最新文本）
async function commitName(row: Suggestion) {
  const r = await api('adjust_ai_suggestion', taskId.value, row.id, row.name, row.ocr)
  if (r && r.ok && r.suggestion) {
    row.name = String(r.suggestion.name || row.name)
    row.ocr = String(r.suggestion.ocr || row.ocr)
  }
}

async function applyIds(ids: number[] | null) {
  const want = ids === null ? null : (ids.length ? ids : null)
  if (ids !== null && ids.length === 0) return
  const r = await api('apply_ai_suggestions', taskId.value, want)
  if (!r || !r.ok) { showToast('应用失败'); return }
  const applied = r.applied || 0
  if (applied > 0) {
    const keys = want === null ? rows.value.map(x => x.id) : want
    const set = new Set(keys)
    rows.value = rows.value.filter(x => !set.has(x.id))
    selected.value = new Set(rows.value.map(x => x.id))
    showToast(`已应用 ${applied} 条`)
    emit('applied')
  } else {
    showToast('没有可应用的建议')
  }
  if (!rows.value.length) close()
}

async function discardIds(ids: number[] | null) {
  const r = await api('discard_ai_suggestions', taskId.value, ids)
  if (!r || !r.ok) { showToast('丢弃失败'); return }
  const dropped = ids ? new Set(ids) : new Set(rows.value.map(x => x.id))
  rows.value = rows.value.filter(x => !dropped.has(x.id))
  selected.value = new Set(rows.value.filter(x => selected.value.has(x.id)).map(x => x.id))
  showToast(`已丢弃 ${r.discarded || dropped.size} 条`)
  if (!rows.value.length) close()
}

function applySelected() { applyIds(Array.from(selected.value)) }
function applyAll() { applyIds(null) }
function discardSelected() { discardIds(Array.from(selected.value)) }
function discardAll() { discardIds(null) }

function background() {
  visible.value = false
}

function close() {
  visible.value = false
  stopPolling()
  running.value = false
}

async function openSettings() {
  try { await api('open_settings') } catch (_) {}
}

onUnmounted(stopPolling)

defineExpose({ open, close })
</script>

<template>
  <div v-if="visible" class="ai-overlay" @click.self="background">
    <div class="ai-box">
      <div class="ai-head">
        <div class="ai-title">AI 标注建议</div>
        <span class="ai-scope" v-if="scopeCount">仅标注选中的 {{ scopeCount }} 张</span>
        <span class="ai-count" v-else-if="rows.length && !running">共 {{ rows.length }} 条</span>
        <span class="spacer"></span>
        <button class="btn btn-ghost btn-sm" @click="background">后台</button>
        <button class="btn btn-ghost btn-sm" @click="close">关闭</button>
      </div>

      <!-- 进度区 -->
      <template v-if="running">
        <div class="ai-msg">{{ message }}</div>
        <div class="ai-bar-wrap">
          <div class="ai-bar" :style="{ width: pct + '%' }"></div>
        </div>
        <div class="ai-meta">
          <span>{{ pct }}%</span>
          <span v-if="total > 0" style="margin-left:12px">共 {{ total }} 张</span>
        </div>
        <div class="ai-foot">
          <button class="btn btn-danger btn-sm" @click="cancelTask">取消标注</button>
        </div>
      </template>

      <!-- 结果区 -->
      <template v-else>
        <div v-if="errorMsg" class="ai-err">
          <div>{{ errorMsg }}</div>
          <button class="btn btn-primary btn-sm" @click="openSettings">打开设置</button>
        </div>
        <template v-else-if="rows.length">
          <div class="ai-list">
            <div v-for="row in rows" :key="row.id" class="ai-row">
              <input
                type="checkbox"
                class="ai-check"
                :checked="selected.has(row.id)"
                :aria-label="'选择 ' + row.name"
                @change="toggleRow(row.id, ($event.target as HTMLInputElement).checked)"
              >
              <img class="ai-thumb" :src="thumbUrl(row)" :alt="row.name" loading="lazy">
              <div class="ai-fields">
                <input
                  class="ai-input"
                  v-model="row.name"
                  placeholder="显示名"
                  maxlength="60"
                  @blur="commitName(row)"
                  @keydown.enter.prevent="($event.target as HTMLInputElement).blur()"
                >
                <input
                  class="ai-input ai-ocr"
                  v-model="row.ocr"
                  placeholder="图上文字（可空）"
                  maxlength="60"
                  @blur="commitName(row)"
                  @keydown.enter.prevent="($event.target as HTMLInputElement).blur()"
                >
              </div>
              <button
                class="btn btn-ghost btn-sm"
                :aria-label="'丢弃 ' + row.name"
                @click="discardIds([row.id])"
              >丢弃</button>
            </div>
          </div>
          <div class="ai-foot">
            <label class="ai-all">
              <input type="checkbox" :checked="allChecked" @change="toggleAll(($event.target as HTMLInputElement).checked)">
              全选
            </label>
            <span class="spacer"></span>
            <button class="btn btn-ghost btn-sm" @click="discardSelected">丢弃选中</button>
            <button class="btn btn-ghost btn-sm" @click="discardAll">全部丢弃</button>
            <button class="btn btn-primary btn-sm" :disabled="selected.size === 0" @click="applySelected">
              应用选中{{ selected.size ? `（${selected.size}）` : '' }}
            </button>
            <button class="btn btn-primary btn-sm" @click="applyAll">应用全部</button>
          </div>
        </template>
        <div v-else class="ai-empty">
          <div>{{ message || '没有待确认的建议' }}</div>
          <button class="btn btn-primary btn-sm" @click="startNew()">重新标注一批</button>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.ai-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 520;
  animation: fadeIn 0.15s ease;
}

.ai-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  width: 620px;
  max-width: 92vw;
  max-height: 82vh;
  padding: 16px 20px 18px;
  display: flex;
  flex-direction: column;
}

.ai-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.ai-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--fg);
}

.ai-count {
  font-size: 11.5px;
  color: var(--muted);
}

.ai-scope {
  font-size: 11.5px;
  color: var(--accent);
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 1px 6px;
}

.spacer { flex: 1; }

.ai-msg {
  font-size: 12px;
  color: var(--fg-secondary);
  margin-bottom: 10px;
}

.ai-bar-wrap {
  width: 100%;
  height: 8px;
  background: var(--bg);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 6px;
}

.ai-bar {
  height: 100%;
  background: var(--accent);
  border-radius: 4px;
  transition: width 0.3s;
}

.ai-meta {
  font-size: 11.5px;
  color: var(--muted);
  margin-bottom: 12px;
}

.ai-err {
  font-size: 12px;
  color: var(--danger);
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0 4px;
}

.ai-empty {
  font-size: 12px;
  color: var(--fg-secondary);
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 0 4px;
}

.ai-list {
  overflow-y: auto;
  flex: 1;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-right: 4px;
}

.ai-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 8px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.ai-check {
  width: 14px;
  height: 14px;
  accent-color: var(--accent);
  flex-shrink: 0;
}

.ai-thumb {
  width: 48px;
  height: 48px;
  object-fit: contain;
  background: var(--bg);
  border-radius: var(--radius-sm);
  flex-shrink: 0;
}

.ai-fields {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.ai-input {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--fg);
  font-size: 12px;
  padding: 4px 6px;
  outline: none;
}

.ai-input:focus { border-color: var(--accent); }

.ai-ocr { color: var(--fg-secondary); font-size: 11.5px; }

.ai-foot {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
}

.ai-all {
  font-size: 12px;
  color: var(--fg-secondary);
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
</style>
