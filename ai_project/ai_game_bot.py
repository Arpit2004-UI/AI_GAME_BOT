"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         AI GAME BOT - TIC TAC TOE                           ║
║              Minimax · Alpha-Beta Pruning · Decision Tree Visualizer         ║
╚══════════════════════════════════════════════════════════════════════════════╝

ALGORITHMS IMPLEMENTED:
  ► Minimax        — Exhaustive game-tree search (guaranteed optimal play)
  ► Alpha-Beta     — Pruned Minimax (skips branches that can't affect outcome)

FEATURES:
  ► Play as X or O against the AI
  ► 3 Difficulty levels (Easy / Medium / Hard)
  ► Live Decision Tree panel showing nodes evaluated, branches pruned
  ► Confidence score · Think time · Search depth display
  ► Win / Draw / Loss tracker

REQUIREMENTS:
  pip install pygame

RUN:
  python ai_game_bot.py
"""

import pygame
import sys
import math
import time
import random
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
#  CONSTANTS & THEME
# ─────────────────────────────────────────────

W, H = 900, 650

# Dark Cyber palette
BG          = (10,  14,  26)
PANEL       = (16,  22,  40)
PANEL2      = (20,  28,  50)
BORDER      = (35,  48,  80)
ACCENT_TEAL = (0,  210, 190)
ACCENT_RED  = (255,  60,  80)
ACCENT_YLW  = (255, 200,  50)
TEXT_HI     = (220, 230, 255)
TEXT_MED    = (130, 150, 190)
TEXT_DIM    = ( 60,  80, 120)
X_COLOR     = (0,  210, 190)
O_COLOR     = (255,  80, 110)
GRID_COLOR  = ( 40,  58,  95)
BADGE_GOOD  = ( 0, 180,  80)
BADGE_BAD   = (200,  40,  60)

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("AI Game Bot  ·  Minimax + Alpha-Beta Pruning")
clock = pygame.time.Clock()

# Fonts (fall back gracefully)
def load_font(size, bold=False):
    try:
        return pygame.font.SysFont("Consolas", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size)

F_TITLE   = load_font(28, bold=True)
F_HEADING = load_font(16, bold=True)
F_BODY    = load_font(14)
F_SMALL   = load_font(12)
F_HUGE    = load_font(72, bold=True)
F_BADGE   = load_font(13, bold=True)
F_NUM     = load_font(22, bold=True)


# ─────────────────────────────────────────────
#  GAME LOGIC
# ─────────────────────────────────────────────

EMPTY = 0
X_PIECE = 1
O_PIECE = 2

WIN_LINES = [
    [0,1,2],[3,4,5],[6,7,8],   # rows
    [0,3,6],[1,4,7],[2,5,8],   # cols
    [0,4,8],[2,4,6],           # diags
]

def check_winner(board):
    for line in WIN_LINES:
        a,b,c = line
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a], line
    return None, None

def board_full(board):
    return all(c != EMPTY for c in board)

def get_moves(board):
    return [i for i,c in enumerate(board) if c == EMPTY]

def board_terminal(board):
    w, _ = check_winner(board)
    return w is not None or board_full(board)


# ─────────────────────────────────────────────
#  AI — MINIMAX + ALPHA-BETA
# ─────────────────────────────────────────────

@dataclass
class AIStats:
    nodes_evaluated: int = 0
    branches_pruned:  int = 0
    search_depth:     int = 0
    think_ms:         float = 0.0
    confidence:       float = 0.0          # 0-100
    best_move:        int   = -1
    tree_log:         list  = field(default_factory=list)  # [(depth, move, score)]


def minimax(board, depth, is_max, ai_piece, human_piece, stats,
            alpha=-math.inf, beta=math.inf, use_ab=True, max_depth=9):
    stats.nodes_evaluated += 1
    stats.search_depth = max(stats.search_depth, depth)

    winner, _ = check_winner(board)
    if winner == ai_piece:
        return 10 - depth
    if winner == human_piece:
        return depth - 10
    if board_full(board) or depth >= max_depth:
        return 0

    moves = get_moves(board)

    if is_max:
        best = -math.inf
        for mv in moves:
            board[mv] = ai_piece
            score = minimax(board, depth+1, False, ai_piece, human_piece,
                            stats, alpha, beta, use_ab, max_depth)
            board[mv] = EMPTY
            if depth == 0:
                stats.tree_log.append((depth, mv, score))
            if score > best:
                best = score
            alpha = max(alpha, best)
            if use_ab and beta <= alpha:
                stats.branches_pruned += len(moves) - moves.index(mv) - 1
                break
        return best
    else:
        best = math.inf
        for mv in moves:
            board[mv] = human_piece
            score = minimax(board, depth+1, True, ai_piece, human_piece,
                            stats, alpha, beta, use_ab, max_depth)
            board[mv] = EMPTY
            if score < best:
                best = score
            beta = min(beta, best)
            if use_ab and beta <= alpha:
                stats.branches_pruned += len(moves) - moves.index(mv) - 1
                break
        return best


DIFFICULTY_DEPTH = {"Easy": 1, "Medium": 3, "Hard": 9}
ALGORITHMS = ["Minimax", "Alpha-Beta"]

def ai_best_move(board, ai_piece, human_piece, difficulty, algorithm):
    stats = AIStats()
    max_depth = DIFFICULTY_DEPTH[difficulty]
    use_ab = (algorithm == "Alpha-Beta")
    t0 = time.time()

    moves = get_moves(board)
    if not moves:
        return -1, stats

    # Easy mode: occasionally random
    if difficulty == "Easy" and random.random() < 0.5:
        stats.best_move = random.choice(moves)
        stats.think_ms = (time.time()-t0)*1000
        stats.confidence = 30.0
        return stats.best_move, stats

    best_score = -math.inf
    best_mv = moves[0]
    bc = board[:]

    for mv in moves:
        bc[mv] = ai_piece
        score = minimax(bc, 0, False, ai_piece, human_piece, stats,
                        use_ab=use_ab, max_depth=max_depth)
        bc[mv] = EMPTY
        if score > best_score:
            best_score = score
            best_mv = mv

    stats.best_move = best_mv
    stats.think_ms = (time.time()-t0)*1000

    # Confidence heuristic
    if best_score > 0:
        stats.confidence = min(95, 60 + best_score*5)
    elif best_score == 0:
        stats.confidence = 50
    else:
        stats.confidence = max(10, 40 + best_score*5)

    return best_mv, stats


# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────

def rounded_rect(surf, color, rect, r=10, alpha=None):
    if alpha is not None:
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0,0,rect[2],rect[3]), border_radius=r)
        surf.blit(s, (rect[0], rect[1]))
    else:
        pygame.draw.rect(surf, color, rect, border_radius=r)

def draw_text(surf, text, font, color, x, y, center=False, right=False):
    s = font.render(str(text), True, color)
    if center:
        x -= s.get_width()//2
    elif right:
        x -= s.get_width()
    surf.blit(s, (x, y))
    return s.get_width()

def glow_text(surf, text, font, color, x, y, center=False):
    """Draw text with a soft glow halo."""
    s = font.render(str(text), True, color)
    glow = pygame.Surface(s.get_size(), pygame.SRCALPHA)
    glow.blit(s, (0,0))
    dim = pygame.transform.smoothscale(glow, (s.get_width()+8, s.get_height()+8))
    for alpha in [30, 20, 10]:
        dim.set_alpha(alpha)
        if center:
            surf.blit(dim, (x - dim.get_width()//2, y - 4))
        else:
            surf.blit(dim, (x-4, y-4))
    if center:
        surf.blit(s, (x - s.get_width()//2, y))
    else:
        surf.blit(s, (x, y))

def draw_x(surf, cx, cy, size, color, thickness=5):
    off = size//2 - 8
    pygame.draw.line(surf, color, (cx-off, cy-off), (cx+off, cy+off), thickness)
    pygame.draw.line(surf, color, (cx+off, cy-off), (cx-off, cy+off), thickness)

def draw_o(surf, cx, cy, size, color, thickness=5):
    r = size//2 - 10
    pygame.draw.circle(surf, color, (cx,cy), r, thickness)

def badge(surf, text, x, y, color):
    w = F_BADGE.size(text)[0] + 16
    rounded_rect(surf, color, (x, y, w, 22), r=6)
    draw_text(surf, text, F_BADGE, BG, x+8, y+4)


# ─────────────────────────────────────────────
#  SELECTOR WIDGET
# ─────────────────────────────────────────────

class Selector:
    def __init__(self, x, y, w, h, options, default=0, label=""):
        self.rect = pygame.Rect(x,y,w,h)
        self.options = options
        self.idx = default
        self.label = label
        self.hovered = False

    def draw(self, surf):
        col = BORDER if not self.hovered else ACCENT_TEAL
        rounded_rect(surf, PANEL2, self.rect, r=8)
        pygame.draw.rect(surf, col, self.rect, 1, border_radius=8)

        # arrows
        aw = 20
        lx = self.rect.x + 6
        rx = self.rect.right - aw - 2
        cy = self.rect.centery

        # left arrow
        la = pygame.Rect(lx, self.rect.y+2, aw, self.rect.h-4)
        rounded_rect(surf, BORDER, la, r=5)
        pts = [(lx+14,cy-5),(lx+14,cy+5),(lx+6,cy)]
        pygame.draw.polygon(surf, TEXT_MED, pts)

        # right arrow
        ra = pygame.Rect(rx, self.rect.y+2, aw, self.rect.h-4)
        rounded_rect(surf, BORDER, ra, r=5)
        pts2 = [(rx+6,cy-5),(rx+6,cy+5),(rx+14,cy)]
        pygame.draw.polygon(surf, TEXT_MED, pts2)

        # value
        val = self.options[self.idx]
        draw_text(surf, val, F_HEADING, TEXT_HI,
                  self.rect.centerx, self.rect.centery-8, center=True)

        if self.label:
            draw_text(surf, self.label, F_SMALL, TEXT_DIM,
                      self.rect.x, self.rect.y-16)

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx,my = event.pos
            if self.rect.collidepoint(mx,my):
                # left third → prev, right third → next
                third = self.rect.width // 3
                if mx < self.rect.x + third:
                    self.idx = (self.idx - 1) % len(self.options)
                elif mx > self.rect.x + 2*third:
                    self.idx = (self.idx + 1) % len(self.options)
                return True
        return False

    @property
    def value(self):
        return self.options[self.idx]


# ─────────────────────────────────────────────
#  BUTTON WIDGET
# ─────────────────────────────────────────────

class Button:
    def __init__(self, x, y, w, h, text, primary=True):
        self.rect = pygame.Rect(x,y,w,h)
        self.text = text
        self.primary = primary
        self.hovered = False
        self._anim = 0.0   # 0→1 for hover glow

    def draw(self, surf):
        if self.hovered:
            self._anim = min(1.0, self._anim + 0.15)
        else:
            self._anim = max(0.0, self._anim - 0.1)

        if self.primary:
            base = ACCENT_TEAL
            glow_col = (0, 255, 230)
        else:
            base = (40, 55, 90)
            glow_col = BORDER

        # glow halo
        if self._anim > 0:
            pad = int(6 * self._anim)
            gr = (self.rect.x-pad, self.rect.y-pad,
                  self.rect.w+pad*2, self.rect.h+pad*2)
            rounded_rect(surf, glow_col, gr, r=12, alpha=int(60*self._anim))

        rounded_rect(surf, base, self.rect, r=8)
        tc = BG if self.primary else TEXT_MED
        draw_text(surf, self.text, F_HEADING, tc,
                  self.rect.centerx, self.rect.centery-8, center=True)

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


# ─────────────────────────────────────────────
#  STAT CARD
# ─────────────────────────────────────────────

def stat_card(surf, x, y, w, h, value, label, color=ACCENT_TEAL):
    rounded_rect(surf, PANEL2, (x,y,w,h), r=8)
    pygame.draw.rect(surf, BORDER, (x,y,w,h), 1, border_radius=8)
    draw_text(surf, value, F_NUM, color, x+w//2, y+8, center=True)
    draw_text(surf, label, F_SMALL, TEXT_DIM, x+w//2, y+h-18, center=True)


# ─────────────────────────────────────────────
#  MAIN GAME CLASS
# ─────────────────────────────────────────────

class AIGameBot:

    GRID_X, GRID_Y = 18, 160
    GRID_W = 220
    CELL = GRID_W // 3

    def __init__(self):
        self.reset_game()
        self.score = {X_PIECE: 0, O_PIECE: 0, "draw": 0}

        # Selectors
        self.sel_player = Selector(18, 100, 130, 40,
                                   ["X", "O"], label="YOU PLAY AS")
        self.sel_diff   = Selector(165, 100, 140, 40,
                                   ["Easy","Medium","Hard"], default=2,
                                   label="DIFFICULTY")
        self.sel_algo   = Selector(320, 100, 170, 40,
                                   ["Minimax","Alpha-Beta"], default=1,
                                   label="ALGORITHM")

        self.btn_new    = Button(18,  590, 120, 36, "NEW GAME")
        self.btn_hint   = Button(150, 590, 80,  36, "HINT", primary=False)

        self.stats = AIStats()
        self.message = "Make a move or let AI go first."
        self.win_line = None
        self.anim_t = 0
        self.hint_cell = -1
        self.thinking = False
        self.ai_move_pending = False
        self.ai_move_timer = 0

    def reset_game(self):
        self.board = [EMPTY]*9
        self.current_player = X_PIECE   # X always starts
        self.game_over = False
        self.winner = None
        self.win_line = None
        self.stats = AIStats()
        self.message = "Make a move or let AI go first."
        self.anim_t = 0
        self.hint_cell = -1
        self.ai_move_pending = False
        self.thinking = False

    @property
    def human_piece(self):
        return X_PIECE if self.sel_player.value == "X" else O_PIECE

    @property
    def ai_piece(self):
        return O_PIECE if self.sel_player.value == "X" else X_PIECE

    def cell_rect(self, i):
        r, c = divmod(i, 3)
        x = self.GRID_X + c * self.CELL
        y = self.GRID_Y + r * self.CELL
        return pygame.Rect(x, y, self.CELL, self.CELL)

    def get_cell_at(self, mx, my):
        for i in range(9):
            if self.cell_rect(i).collidepoint(mx, my):
                return i
        return -1

    def do_move(self, idx, piece):
        if self.board[idx] != EMPTY or self.game_over:
            return
        self.board[idx] = piece
        self.hint_cell = -1
        w, wl = check_winner(self.board)
        if w:
            self.game_over = True
            self.winner = w
            self.win_line = wl
            if w == self.human_piece:
                self.message = "🎉  You win!"
                self.score[self.human_piece] += 1
            else:
                self.message = "🤖  AI wins!"
                self.score[self.ai_piece] += 1
        elif board_full(self.board):
            self.game_over = True
            self.winner = 0
            self.message = "🤝  It's a draw!"
            self.score["draw"] += 1
        else:
            self.current_player = (
                O_PIECE if piece == X_PIECE else X_PIECE
            )
            if self.current_player == self.ai_piece:
                self.ai_move_pending = True
                self.ai_move_timer = pygame.time.get_ticks() + 400
                self.message = "🤔  AI is thinking..."

    def trigger_ai(self):
        mv, self.stats = ai_best_move(
            self.board[:], self.ai_piece, self.human_piece,
            self.sel_diff.value, self.sel_algo.value
        )
        if mv >= 0:
            self.do_move(mv, self.ai_piece)
            if not self.game_over:
                self.message = f"Your turn!  ({self.sel_diff.value} · {self.sel_algo.value})"

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                self.reset_game()

            self.sel_player.handle(ev)
            self.sel_diff.handle(ev)
            self.sel_algo.handle(ev)

            if self.btn_new.handle(ev):
                self.reset_game()

            if self.btn_hint.handle(ev) and not self.game_over:
                if self.current_player == self.human_piece:
                    mv, _ = ai_best_move(
                        self.board[:], self.human_piece, self.ai_piece,
                        "Hard", self.sel_algo.value
                    )
                    self.hint_cell = mv

            if ev.type == pygame.MOUSEBUTTONDOWN:
                mx,my = ev.pos
                idx = self.get_cell_at(mx,my)
                if (idx >= 0 and not self.game_over
                        and self.current_player == self.human_piece
                        and not self.ai_move_pending):
                    self.do_move(idx, self.human_piece)

    def update(self):
        self.anim_t += 0.05
        if self.ai_move_pending:
            if pygame.time.get_ticks() >= self.ai_move_timer:
                self.ai_move_pending = False
                self.trigger_ai()

    # ─── DRAW ───────────────────────────────

    def draw_background(self):
        screen.fill(BG)
        # subtle grid dots
        for x in range(0, W, 40):
            for y in range(0, H, 40):
                pygame.draw.circle(screen, (20,28,50), (x,y), 1)

    def draw_header(self):
        # Title area
        title = "◈  AI GAME BOT"
        glow_text(screen, title, F_TITLE, ACCENT_TEAL, W//2, 18, center=True)
        sub = "Minimax  ·  Alpha-Beta Pruning  ·  Decision Tree Visualizer"
        draw_text(screen, sub, F_SMALL, TEXT_DIM, W//2, 52, center=True)
        pygame.draw.line(screen, BORDER, (18,72),(W-18,72),1)

    def draw_selectors(self):
        self.sel_player.draw(screen)
        self.sel_diff.draw(screen)
        self.sel_algo.draw(screen)

    def draw_board(self):
        gx, gy = self.GRID_X, self.GRID_Y
        gw = self.GRID_W
        cell = self.CELL

        # Board background
        rounded_rect(screen, PANEL, (gx-4, gy-4, gw+8, gw+8), r=12)
        pygame.draw.rect(screen, BORDER, (gx-4, gy-4, gw+8, gw+8), 1,
                         border_radius=12)

        # Grid lines
        for i in range(1,3):
            pygame.draw.line(screen, GRID_COLOR,
                             (gx+i*cell, gy), (gx+i*cell, gy+gw), 2)
            pygame.draw.line(screen, GRID_COLOR,
                             (gx, gy+i*cell), (gx+gw, gy+i*cell), 2)

        # Cells
        for idx in range(9):
            r,c = divmod(idx,3)
            cx = gx + c*cell + cell//2
            cy = gy + r*cell + cell//2
            rc = self.cell_rect(idx)

            # Hover highlight
            mx,my = pygame.mouse.get_pos()
            if (rc.collidepoint(mx,my)
                    and self.board[idx] == EMPTY
                    and not self.game_over
                    and self.current_player == self.human_piece):
                rounded_rect(screen, ACCENT_TEAL, rc, r=6, alpha=25)

            # Hint highlight
            if idx == self.hint_cell and self.board[idx] == EMPTY:
                rounded_rect(screen, ACCENT_YLW, rc, r=6, alpha=40)
                draw_text(screen, "?", F_HEADING, ACCENT_YLW,
                          cx, cy-10, center=True)

            # Pieces
            if self.board[idx] == X_PIECE:
                draw_x(screen, cx, cy, cell, X_COLOR, thickness=4)
            elif self.board[idx] == O_PIECE:
                draw_o(screen, cx, cy, cell, O_COLOR, thickness=4)

        # Win line
        if self.win_line:
            a,b = self.win_line[0], self.win_line[2]
            ra,ca = divmod(a,3)
            rb,cb = divmod(b,3)
            x1 = gx + ca*cell + cell//2
            y1 = gy + ra*cell + cell//2
            x2 = gx + cb*cell + cell//2
            y2 = gy + rb*cell + cell//2
            col = X_COLOR if self.winner == X_PIECE else O_COLOR
            pygame.draw.line(screen, col, (x1,y1),(x2,y2), 5)

        # Game over overlay
        if self.game_over:
            rounded_rect(screen, BG, (gx-4, gy-4, gw+8, gw+8), r=12, alpha=160)
            if self.winner == 0:
                msg, col = "DRAW", ACCENT_YLW
            elif self.winner == self.human_piece:
                msg, col = "WIN!", ACCENT_TEAL
            else:
                msg, col = "LOSE", ACCENT_RED
            glow_text(screen, msg, F_HUGE, col, gx+gw//2, gy+gw//2-30, center=True)

    def draw_status_bar(self):
        sx, sy = self.GRID_X, self.GRID_Y + self.GRID_W + 10
        bw = (self.GRID_W - 8) // 3
        bh = 28

        you_col  = BADGE_GOOD
        draw_col = (70, 90, 130)
        ai_col   = BADGE_BAD

        rounded_rect(screen, PANEL2, (sx, sy, self.GRID_W, 36), r=8)

        # You badge
        badge(screen, f"You  {self.score[self.human_piece]}", sx+4, sy+7, you_col)
        # Draw badge
        dw = F_BADGE.size(f"Draw  {self.score['draw']}")[0] + 16
        draw_text(screen, f"Draws  {self.score['draw']}", F_BADGE, TEXT_MED,
                  sx + self.GRID_W//2, sy+11, center=True)
        # AI badge
        ai_txt = f"AI  {self.score[self.ai_piece]}"
        aiw = F_BADGE.size(ai_txt)[0] + 16
        badge(screen, ai_txt, sx + self.GRID_W - aiw - 4, sy+7, ai_col)

    def draw_message_bar(self):
        bx, by = self.GRID_X, self.GRID_Y - 52
        rounded_rect(screen, PANEL2, (bx, by, self.GRID_W, 42), r=8)
        pygame.draw.rect(screen, BORDER, (bx, by, self.GRID_W, 42), 1,
                         border_radius=8)
        draw_text(screen, "Your turn!" if not self.game_over else "Game Over",
                  F_HEADING, TEXT_HI, bx+12, by+7)
        status = self.message
        col = ACCENT_TEAL if "win" in status.lower() else (
              ACCENT_RED if "AI wins" in status else TEXT_MED)
        draw_text(screen, status, F_SMALL, col, bx+12, by+26)

        # Current piece indicator
        cp = self.current_player
        cx2 = bx + self.GRID_W - 20
        cy2 = by + 21
        if not self.game_over:
            if cp == X_PIECE:
                draw_x(screen, cx2, cy2, 28, X_COLOR, 2)
            else:
                draw_o(screen, cx2, cy2, 28, O_COLOR, 2)

    def draw_decision_tree_panel(self):
        px, py = 260, 80
        pw, ph = W - px - 18, H - py - 18
        rounded_rect(screen, PANEL, (px,py,pw,ph), r=12)
        pygame.draw.rect(screen, BORDER, (px,py,pw,ph), 1, border_radius=12)

        # Title bar
        rounded_rect(screen, PANEL2, (px,py,pw,36), r=12)
        draw_text(screen, "DECISION TREE", F_HEADING, TEXT_HI, px+14, py+11)
        algo_txt = self.sel_algo.value.upper()
        badge(screen, algo_txt, px+pw-80, py+9,
              ACCENT_TEAL if "Beta" in self.sel_algo.value else ACCENT_YLW)

        # Stat cards row 1
        cw, ch = (pw-32)//2 - 4, 54
        cy1 = py + 50
        stat_card(screen, px+8, cy1, cw, ch,
                  str(self.stats.nodes_evaluated), "nodes eval'd", ACCENT_TEAL)
        stat_card(screen, px+cw+16, cy1, cw, ch,
                  str(self.stats.branches_pruned), "branches pruned", ACCENT_RED)

        # Stat cards row 2
        cy2 = cy1 + ch + 8
        stat_card(screen, px+8, cy2, cw, ch,
                  str(self.stats.search_depth), "search depth", ACCENT_YLW)
        think = f"{self.stats.think_ms:.0f}ms" if self.stats.think_ms < 1000 else \
                f"{self.stats.think_ms/1000:.2f}s"
        stat_card(screen, px+cw+16, cy2, cw, ch,
                  think, "think time", TEXT_MED)

        # Decision tree visualization
        tree_y = cy2 + ch + 16
        tree_h = py + ph - tree_y - 55
        rounded_rect(screen, PANEL2, (px+8, tree_y, pw-16, tree_h), r=8)
        pygame.draw.rect(screen, BORDER, (px+8, tree_y, pw-16, tree_h), 1,
                         border_radius=8)

        if self.stats.tree_log:
            self.draw_tree_graph(px+8, tree_y, pw-16, tree_h)
        else:
            lines = [
                "Make a move or",
                "let AI go first.",
                "",
                "The decision tree",
                "will appear here.",
            ]
            for i,ln in enumerate(lines):
                col = TEXT_MED if ln else TEXT_DIM
                draw_text(screen, ln, F_BODY, col,
                          px+pw//2, tree_y+14+i*20, center=True)

        # Confidence bar
        conf_y = py + ph - 46
        draw_text(screen, "AI confidence", F_SMALL, TEXT_DIM, px+12, conf_y)
        conf_pct = self.stats.confidence / 100
        bar_x, bar_y = px+12, conf_y+16
        bar_w = pw - 24
        rounded_rect(screen, PANEL2, (bar_x, bar_y, bar_w, 12), r=6)
        fill_w = int(bar_w * conf_pct)
        if fill_w > 0:
            col = ACCENT_TEAL if conf_pct>0.6 else (
                  ACCENT_YLW if conf_pct>0.4 else ACCENT_RED)
            rounded_rect(screen, col, (bar_x, bar_y, fill_w, 12), r=6)
        pct_txt = f"{self.stats.confidence:.0f}%"
        draw_text(screen, pct_txt, F_SMALL, TEXT_MED,
                  bar_x + bar_w + 6, bar_y - 1)

    def draw_tree_graph(self, x, y, w, h):
        """Visual bar chart of top-level move scores from tree_log."""
        log = self.stats.tree_log
        if not log:
            return

        n = len(log)
        margin = 10
        bar_area_w = w - 2*margin
        bar_w = max(8, bar_area_w // n - 4)
        max_h = h - 50

        scores = [s for (_, _, s) in log]
        s_min = min(scores) if scores else -1
        s_max = max(scores) if scores else 1
        rng = max(1, s_max - s_min)

        draw_text(screen, "Move scores (root depth)", F_SMALL, TEXT_DIM,
                  x+w//2, y+6, center=True)

        for i,(_, mv, sc) in enumerate(log):
            bx = x + margin + i*(bar_w+4)
            norm = (sc - s_min) / rng
            bh = max(4, int(norm * max_h * 0.8))
            by = y + h - 30 - bh

            is_best = (mv == self.stats.best_move)
            col = ACCENT_TEAL if is_best else (
                  ACCENT_RED if sc < 0 else TEXT_DIM)

            rounded_rect(screen, col, (bx, by, bar_w, bh), r=3)

            # score label
            sc_str = f"{sc:+d}" if sc != 0 else "0"
            draw_text(screen, sc_str, F_SMALL, col, bx+bar_w//2, by-14, center=True)

            # cell label
            row, col_idx = divmod(mv, 3)
            cell_lbl = f"R{row+1}C{col_idx+1}"
            draw_text(screen, cell_lbl, F_SMALL, TEXT_DIM,
                      bx+bar_w//2, y+h-24, center=True)

        # legend
        draw_text(screen, "★=best", F_SMALL, ACCENT_TEAL, x+margin, y+h-14)

    def draw_buttons(self):
        self.btn_new.draw(screen)
        self.btn_hint.draw(screen)
        draw_text(screen, "Press R to restart",
                  F_SMALL, TEXT_DIM, 250, 600)

    def draw(self):
        self.draw_background()
        self.draw_header()
        self.draw_selectors()
        self.draw_message_bar()
        self.draw_board()
        self.draw_status_bar()
        self.draw_decision_tree_panel()
        self.draw_buttons()

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            clock.tick(60)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(__doc__)
    bot = AIGameBot()
    bot.run()
