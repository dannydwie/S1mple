#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

static constexpr int INF = 30000;
static constexpr int MATE = 29000;

enum { EMPTY=0, W=0, B=1 };
enum { PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6 };

struct Move {
    int from=0,to=0,piece=0,capture=0,promo=0,flags=0,score=0;

    bool operator==(const Move& x) const {
        return from==x.from && to==x.to && promo==x.promo;
    }
};

enum { EP=1, CASTLE=2, DOUBLE=4 };

static int board[128];
static int side;
static int castleRights;
static int epSq;
static int halfmove;
static int fullmove;

static uint64_t zob[2][7][64];
static uint64_t zobSide;
static uint64_t zobCastle[16];
static uint64_t zobEp[64];
static bool zobInit=false;

static const int pieceVal[7]={
    0,100,320,330,500,900,0
};

static const int knightD[8]={
    -33,-31,-18,-14,14,18,31,33
};

static const int bishopD[4]={
    -17,-15,15,17
};

static const int rookD[4]={
    -16,-1,1,16
};

static const int kingD[8]={
    -17,-16,-15,-1,1,15,16,17
};

struct Undo {
    int captured;
    int castle;
    int ep;
    int half;
    int full;
    uint64_t key;
};

struct TTEntry {
    uint64_t key=0;
    int depth=-1;
    int score=0;
    int flag=0;
    Move best{};
};

enum {
    TT_EXACT,
    TT_ALPHA,
    TT_BETA
};

static vector<TTEntry> tt(1<<20);

static uint64_t hashKey=0;
static int nodes=0;

static chrono::steady_clock::time_point stopAt;
static atomic<bool> stopSearch(false);

static uint64_t rng64() {
    static uint64_t x=0x9e3779b97f4a7c15ULL;

    x^=x>>12;
    x^=x<<25;
    x^=x>>27;

    return x*2685821657736338717ULL;
}

static int sq64(int s) {
    return (s&7)+((s>>4)<<3);
}

static int fileOf(int s) {
    return s&7;
}

static int rankOf(int s) {
    return s>>4;
}

static int colorOf(int p) {
    return p>0 ? W : B;
}

static int pc(int p) {
    return p>0 ? p : -p;
}

static void initZob() {

    if(zobInit)
        return;

    for(int c=0;c<2;c++)
        for(int p=1;p<=6;p++)
            for(int s=0;s<64;s++)
                zob[c][p][s]=rng64();

    for(auto &x:zobCastle)
        x=rng64();

    for(auto &x:zobEp)
        x=rng64();

    zobSide=rng64();

    zobInit=true;
}

static void recomputeHash() {

    hashKey=0;

    for(int s=0;s<128;s++) {

        if(s&8)
            continue;

        int p=board[s];

        if(!p)
            continue;

        hashKey^=
            zob[colorOf(p)]
               [pc(p)]
               [sq64(s)];
    }

    hashKey^=zobCastle[castleRights];

    if(epSq>=0)
        hashKey^=zobEp[sq64(epSq)];

    if(side==B)
        hashKey^=zobSide;
}

static void clearBoard() {

    fill(begin(board),end(board),0);

    side=W;
    castleRights=0;
    epSq=-1;
    halfmove=0;
    fullmove=1;
}

static int strSq(const string& s) {

    if(s.size()!=2)
        return -1;

    int f=s[0]-'a';
    int r=s[1]-'1';

    if(f<0 || f>7 || r<0 || r>7)
        return -1;

    return r*16+f;
}

static string sqStr(int s) {

    string r="a1";

    r[0]='a'+fileOf(s);
    r[1]='1'+rankOf(s);

    return r;
}

static void setStart() {

    clearBoard();

    const string back="RNBQKBNR";

    for(int f=0;f<8;f++) {

        board[f]=back[f]-'A'+1;
        board[16+f]=PAWN;

        board[96+f]=-PAWN;
        board[112+f]=-(back[f]-'A'+1);
    }

    castleRights=15;
    epSq=-1;

    recomputeHash();
}

static void setFEN(string fen) {

    clearBoard();

    stringstream ss(fen);

    string placement;
    string stm;
    string cr;
    string eps;

    ss>>placement>>stm>>cr>>eps
      >>halfmove>>fullmove;

    int s=112;

    for(char c:placement) {

        if(c=='/') {
            s-=24;
            continue;
        }

        if(isdigit((unsigned char)c)) {
            s+=c-'0';
            continue;
        }

        int p=1;

        string u="PNBRQK";

        size_t k=
            u.find(toupper((unsigned char)c));

        if(k!=string::npos)
            p=(int)k+1;

        board[s++]=
            isupper((unsigned char)c)
            ? p
            : -p;
    }

    side=(stm=="b") ? B : W;

    castleRights=0;

    if(cr.find('K')!=string::npos)
        castleRights|=1;

    if(cr.find('Q')!=string::npos)
        castleRights|=2;

    if(cr.find('k')!=string::npos)
        castleRights|=4;

    if(cr.find('q')!=string::npos)
        castleRights|=8;

    epSq=
        eps=="-"
        ? -1
        : strSq(eps);

    recomputeHash();
}

static int kingSquare(int c) {

    int wanted=
        c==W
        ? KING
        : -KING;

    for(int s=0;s<128;s++) {

        if(s&8)
            continue;

        if(board[s]==wanted)
            return s;
    }

    return -1;
}

static bool attacked(int s,int by) {

    int pawn=
        by==W
        ? PAWN
        : -PAWN;

    int f=fileOf(s);

    if(by==W) {

        if(f && s>=16 && board[s-17]==pawn)
            return true;

        if(f<7 && s>=16 && board[s-15]==pawn)
            return true;

    } else {

        if(f && s<112 && board[s+15]==pawn)
            return true;

        if(f<7 && s<112 && board[s+17]==pawn)
            return true;
    }

    int n=
        by==W
        ? KNIGHT
        : -KNIGHT;

    for(int d:knightD) {

        int x=s+d;

        if(!(x&8) && board[x]==n)
            return true;
    }

    int bishop=
        by==W
        ? BISHOP
        : -BISHOP;

    int queen=
        by==W
        ? QUEEN
        : -QUEEN;

    for(int d:bishopD) {

        for(int x=s+d;!(x&8);x+=d) {

            if(board[x]) {

                if(board[x]==bishop ||
                   board[x]==queen)
                    return true;

                break;
            }
        }
    }

    int rook=
        by==W
        ? ROOK
        : -ROOK;

    for(int d:rookD) {

        for(int x=s+d;!(x&8);x+=d) {

            if(board[x]) {

                if(board[x]==rook ||
                   board[x]==queen)
                    return true;

                break;
            }
        }
    }

    int king=
        by==W
        ? KING
        : -KING;

    for(int d:kingD) {

        int x=s+d;

        if(!(x&8) && board[x]==king)
            return true;
    }

    return false;
}

static bool inCheck(int c) {

    int k=kingSquare(c);

    return k>=0 &&
           attacked(k,c^1);
}

static void addMove(
    vector<Move>& v,
    int from,
    int to,
    int flags=0,
    int promo=0
) {

    Move m;

    m.from=from;
    m.to=to;
    m.piece=board[from];
    m.capture=board[to];
    m.flags=flags;
    m.promo=promo;

    v.push_back(m);
}

static void genPseudo(
    vector<Move>& v,
    bool capturesOnly=false
) {

    v.clear();

    for(int s=0;s<128;s++) {

        if(s&8)
            continue;

        int p=board[s];

        if(!p || colorOf(p)!=side)
            continue;

        int t=pc(p);

        if(t==PAWN) {

            int d=
                side==W
                ? 16
                : -16;

            int start=
                side==W
                ? 1
                : 6;

            int prom=
                side==W
                ? 6
                : 1;

            int one=s+d;

            if(!capturesOnly &&
               !(one&8) &&
               !board[one]) {

                if(rankOf(one)==prom) {

                    for(int q:
                        {QUEEN,ROOK,BISHOP,KNIGHT})

                        addMove(
                            v,s,one,0,q
                        );

                } else {

                    addMove(v,s,one);

                    int two=s+2*d;

                    if(rankOf(s)==start &&
                       !board[two])

                        addMove(
                            v,s,two,DOUBLE
                        );
                }
            }

            for(int df:{-1,1}) {

                int x=s+d+df;

                if(x&8)
                    continue;

                if(board[x] &&
                   colorOf(board[x])!=side) {

                    if(rankOf(x)==prom) {

                        for(int q:
                            {QUEEN,ROOK,BISHOP,KNIGHT})

                            addMove(
                                v,s,x,0,q
                            );

                    } else {

                        addMove(v,s,x);
                    }

                } else if(x==epSq) {

                    addMove(
                        v,s,x,EP
                    );
                }
            }

        } else if(t==KNIGHT) {

            for(int d:knightD) {

                int x=s+d;

                if(x&8)
                    continue;

                if(!board[x] ||
                   colorOf(board[x])!=side)

                    if(!capturesOnly || board[x])
                        addMove(v,s,x);
            }

        } else if(
            t==BISHOP ||
            t==ROOK ||
            t==QUEEN
        ) {

            int dirs[8];
            int n=0;

            if(t==BISHOP ||
               t==QUEEN) {

                for(int d:bishopD)
                    dirs[n++]=d;
            }

            if(t==ROOK ||
               t==QUEEN) {

                for(int d:rookD)
                    dirs[n++]=d;
            }

            for(int i=0;i<n;i++) {

                int d=dirs[i];

                for(
                    int x=s+d;
                    !(x&8);
                    x+=d
                ) {

                    if(!board[x]) {

                        if(!capturesOnly)
                            addMove(v,s,x);

                    } else {

                        if(colorOf(board[x])!=side)
                            addMove(v,s,x);

                        break;
                    }
                }
            }

        } else if(t==KING) {

            for(int d:kingD) {

                int x=s+d;

                if(x&8)
                    continue;

                if(!board[x] ||
                   colorOf(board[x])!=side)

                    if(!capturesOnly || board[x])
                        addMove(v,s,x);
            }

            if(!capturesOnly &&
               !inCheck(side)) {

                if(
                    side==W &&
                    (castleRights&1) &&
                    !board[5] &&
                    !board[6] &&
                    !attacked(5,B) &&
                    !attacked(6,B)
                )
                    addMove(v,4,6,CASTLE);

                if(
                    side==W &&
                    (castleRights&2) &&
                    !board[3] &&
                    !board[2] &&
                    !board[1] &&
                    !attacked(3,B) &&
                    !attacked(2,B)
                )
                    addMove(v,4,2,CASTLE);

                if(
                    side==B &&
                    (castleRights&4) &&
                    !board[117] &&
                    !board[118] &&
                    !attacked(117,W) &&
                    !attacked(118,W)
                )
                    addMove(v,116,118,CASTLE);

                if(
                    side==B &&
                    (castleRights&8) &&
                    !board[115] &&
                    !board[114] &&
                    !board[113] &&
                    !attacked(115,W) &&
                    !attacked(114,W)
                )
                    addMove(v,116,114,CASTLE);
            }
        }
    }
}

static bool makeMove(
    const Move& m,
    Undo& u
) {

    u.captured=board[m.to];
    u.castle=castleRights;
    u.ep=epSq;
    u.half=halfmove;
    u.full=fullmove;
    u.key=hashKey;

    int p=board[m.from];
    int c=side;

    board[m.from]=0;
    board[m.to]=p;

    if(m.flags&EP)
        board[
            m.to+(c==W?-16:16)
        ]=0;

    if(m.promo)
        board[m.to]=
            c==W
            ? m.promo
            : -m.promo;

    if(m.flags&CASTLE) {

        if(m.to==6) {
            board[5]=board[7];
            board[7]=0;
        }

        else if(m.to==2) {
            board[3]=board[0];
            board[0]=0;
        }

        else if(m.to==118) {
            board[117]=board[119];
            board[119]=0;
        }

        else if(m.to==114) {
            board[115]=board[112];
            board[112]=0;
        }
    }

    if(p==KING || p==-KING) {

        if(c==W)
            castleRights&=~3;
        else
            castleRights&=~12;
    }

    if(m.from==0 || m.to==0)
        castleRights&=~2;

    if(m.from==7 || m.to==7)
        castleRights&=~1;

    if(m.from==112 || m.to==112)
        castleRights&=~8;

    if(m.from==119 || m.to==119)
        castleRights&=~4;

    epSq=-1;

    if(m.flags&DOUBLE)
        epSq=
            m.from+
            (c==W?16:-16);

    halfmove=
        pc(p)==PAWN ||
        u.captured ||
        (m.flags&EP)
        ? 0
        : halfmove+1;

    if(c==B)
        fullmove++;

    side^=1;

    recomputeHash();

    if(inCheck(c)) {

        // Restore immediately.
        side^=1;

        castleRights=u.castle;
        epSq=u.ep;
        halfmove=u.half;
        fullmove=u.full;

        board[m.from]=
            m.promo
            ? (side==W?PAWN:-PAWN)
            : p;

        board[m.to]=u.captured;

        if(m.flags&EP)
            board[
                m.to+(side==W?-16:16)
            ]=
                side==W
                ? -PAWN
                : PAWN;

        if(m.flags&CASTLE) {

            if(m.to==6) {
                board[7]=board[5];
                board[5]=0;
            }

            else if(m.to==2) {
                board[0]=board[3];
                board[3]=0;
            }

            else if(m.to==118) {
                board[119]=board[117];
                board[117]=0;
            }

            else if(m.to==114) {
                board[112]=board[115];
                board[115]=0;
            }
        }

        hashKey=u.key;

        return false;
    }

    return true;
}

static void undoMove(
    const Move& m,
    const Undo& u
) {

    side^=1;

    castleRights=u.castle;
    epSq=u.ep;
    halfmove=u.half;
    fullmove=u.full;

    int p=board[m.to];

    if(m.promo)
        p=
            side==W
            ? PAWN
            : -PAWN;

    board[m.from]=p;
    board[m.to]=u.captured;

    if(m.flags&EP)
        board[
            m.to+(side==W?-16:16)
        ]=
            side==W
            ? -PAWN
            : PAWN;

    if(m.flags&CASTLE) {

        if(m.to==6) {
            board[7]=board[5];
            board[5]=0;
        }

        else if(m.to==2) {
            board[0]=board[3];
            board[3]=0;
        }

        else if(m.to==118) {
            board[119]=board[117];
            board[117]=0;
        }

        else if(m.to==114) {
            board[112]=board[115];
            board[115]=0;
        }
    }

    hashKey=u.key;
}

static vector<Move> legalMoves(
    bool capturesOnly=false
) {

    vector<Move> pseudo;
    vector<Move> legal;

    genPseudo(
        pseudo,
        capturesOnly
    );

    for(auto &m:pseudo) {

        Undo u;

        if(makeMove(m,u)) {

            legal.push_back(m);

            undoMove(m,u);
        }
    }

    return legal;
}

/* =========================
   SIMPLELOGICS EVALUATION
   ========================= */

static const int pawnTable[64]={
     0,  5,  5, -5, -5, 10, 10,  0,
     0, 10, -5,  0,  0, -5, 10,  0,
     0,  5, 10, 20, 20, 10,  5,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    20, 20, 30, 40, 40, 30, 20, 20,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0
};

static const int knightTable[64]={
   -30,-20,-10,-10,-10,-10,-20,-30,
   -20, -5,  0,  5,  5,  0, -5,-20,
   -10,  5, 10, 15, 15, 10,  5,-10,
   -10,  0, 15, 20, 20, 15,  0,-10,
   -10,  5, 15, 20, 20, 15,  5,-10,
   -10,  0, 10, 15, 15, 10,  0,-10,
   -20, -5,  0,  0,  0,  0, -5,-20,
   -30,-20,-10,-10,-10,-10,-20,-30
};

static const int bishopTable[64]={
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};

static const int kingTable[64]={
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    30, 30, 10,  0,  0, 10, 30, 30,
    30, 40, 20,  0,  0, 20, 40, 30,
    30, 30, 20,  0,  0, 20, 30, 30
};

static int mirrorSq(int s) {

    return
        (7-rankOf(s))*8+
        fileOf(s);
}

static int evaluate() {

    int score=0;

    int whiteBishops=0;
    int blackBishops=0;

    for(int s=0;s<128;s++) {

        if(s&8)
            continue;

        int p=board[s];

        if(!p)
            continue;

        int c=colorOf(p);
        int t=pc(p);

        int q=sq64(s);

        int v=pieceVal[t];

        int psq=
            c==W
            ? q
            : mirrorSq(s);

        if(t==PAWN)
            v+=pawnTable[psq];

        if(t==KNIGHT)
            v+=knightTable[psq];

        if(t==BISHOP)
            v+=bishopTable[psq];

        if(t==KING)
            v+=kingTable[psq];

        if(t==BISHOP) {

            if(c==W)
                whiteBishops++;
            else
                blackBishops++;
        }

        score+=
            c==W
            ? v
            : -v;
    }

    if(whiteBishops>=2)
        score+=35;

    if(blackBishops>=2)
        score-=35;

    int oldSide=side;

    vector<Move> a;

    genPseudo(a,false);

    int mobWhiteOrSide=a.size();

    side^=1;

    vector<Move> b2;

    genPseudo(b2,false);

    int mobOther=b2.size();

    side=oldSide;

    score+=
        oldSide==W
        ? (mobWhiteOrSide-mobOther)*2
        : (mobOther-mobWhiteOrSide)*2;

    return
        side==W
        ? score
        : -score;
}

/* =========================
   SEARCH
   ========================= */

static bool timeUp() {

    if((nodes&2047)!=0)
        return false;

    return
        stopSearch ||
        chrono::steady_clock::now()>=stopAt;
}

static int moveScore(
    const Move& m,
    const Move* ttMove=nullptr
) {

    int s=0;

    if(ttMove && m==*ttMove)
        s+=1000000;

    if(m.capture) {

        s+=
            10000+
            10*pieceVal[pc(m.capture)]-
            pieceVal[pc(m.piece)];
    }

    if(m.promo)
        s+=8000+pieceVal[m.promo];

    return s;
}

static void orderMoves(
    vector<Move>& moves,
    const Move* ttMove=nullptr
) {

    for(auto &m:moves)
        m.score=
            moveScore(m,ttMove);

    stable_sort(
        moves.begin(),
        moves.end(),
        [](const Move&a,const Move&b){
            return a.score>b.score;
        }
    );
}

static int qsearch(
    int alpha,
    int beta,
    int ply
) {

    nodes++;

    if(timeUp())
        return 0;

    int stand=evaluate();

    if(stand>=beta)
        return beta;

    if(stand>alpha)
        alpha=stand;

    vector<Move> moves=
        legalMoves(true);

    orderMoves(moves);

    for(auto &m:moves) {

        Undo u;

        if(!makeMove(m,u))
            continue;

        int score=
            -qsearch(
                -beta,
                -alpha,
                ply+1
            );

        undoMove(m,u);

        if(stopSearch)
            return 0;

        if(score>=beta)
            return beta;

        if(score>alpha)
            alpha=score;
    }

    return alpha;
}

static int search(
    int depth,
    int alpha,
    int beta,
    int ply
) {

    if(timeUp())
        return 0;

    nodes++;

    bool check=inCheck(side);

    if(depth<=0)
        return qsearch(
            alpha,
            beta,
            ply
        );

    if(ply>0 && check)
        depth++;

    size_t idx=
        hashKey&
        (tt.size()-1);

    TTEntry &entry=tt[idx];

    Move ttMove=entry.best;

    if(
        entry.key==hashKey &&
        entry.depth>=depth
    ) {

        if(entry.flag==TT_EXACT)
            return entry.score;

        if(
            entry.flag==TT_ALPHA &&
            entry.score<=alpha
        )
            return alpha;

        if(
            entry.flag==TT_BETA &&
            entry.score>=beta
        )
            return beta;
    }

    vector<Move> moves=
        legalMoves(false);

    if(moves.empty()) {

        if(check)
            return -MATE+ply;

        return 0;
    }

    orderMoves(
        moves,
        entry.key==hashKey
        ? &ttMove
        : nullptr
    );

    int best=-INF;

    int originalAlpha=alpha;

    Move bestMove=moves[0];

    int moveIndex=0;

    for(auto &m:moves) {

        Undo u;

        if(!makeMove(m,u))
            continue;

        int newDepth=depth-1;

        /*
           Late Move Reduction:
           Quiet late moves are searched
           one ply shallower.
        */

        if(
            moveIndex>=4 &&
            depth>=3 &&
            !m.capture &&
            !m.promo &&
            !check
        )
            newDepth=
                max(1,newDepth-1);

        int score=
            -search(
                newDepth,
                -beta,
                -alpha,
                ply+1
            );

        undoMove(m,u);

        if(stopSearch)
            return 0;

        if(score>best) {

            best=score;
            bestMove=m;
        }

        if(score>alpha)
            alpha=score;

        if(alpha>=beta)
            break;

        moveIndex++;
    }

    entry.key=hashKey;
    entry.depth=depth;
    entry.score=best;
    entry.best=bestMove;

    if(best<=originalAlpha)
        entry.flag=TT_ALPHA;

    else if(best>=beta)
        entry.flag=TT_BETA;

    else
        entry.flag=TT_EXACT;

    return best;
}

/* =========================
   UCI
   ========================= */

static string moveToUci(
    const Move&m
) {

    string s=
        sqStr(m.from)+
        sqStr(m.to);

    if(m.promo) {

        char c='q';

        if(m.promo==KNIGHT)
            c='n';

        else if(m.promo==BISHOP)
            c='b';

        else if(m.promo==ROOK)
            c='r';

        s+=c;
    }

    return s;
}

static Move parseUci(
    const string& u
) {

    Move result{};

    if(u.size()<4)
        return result;

    result.from=
        strSq(u.substr(0,2));

    result.to=
        strSq(u.substr(2,2));

    if(u.size()>4) {

        char c=
            tolower(
                (unsigned char)u[4]
            );

        if(c=='q')
            result.promo=QUEEN;

        else if(c=='r')
            result.promo=ROOK;

        else if(c=='b')
            result.promo=BISHOP;

        else
            result.promo=KNIGHT;
    }

    vector<Move> moves=
        legalMoves(false);

    for(auto &m:moves) {

        if(
            m.from==result.from &&
            m.to==result.to &&
            m.promo==result.promo
        )
            return m;
    }

    return result;
}

static Move think(
    int milliseconds
) {

    stopSearch=false;
    nodes=0;

    stopAt=
        chrono::steady_clock::now()+
        chrono::milliseconds(
            max(50,milliseconds)
        );

    Move best{};

    for(int depth=1;depth<=64;depth++) {

        int score=
            search(
                depth,
                -INF,
                INF,
                0
            );

        if(stopSearch)
            break;

        auto &entry=
            tt[
                hashKey&
                (tt.size()-1)
            ];

        if(entry.key==hashKey)
            best=entry.best;

        cout
            <<"info depth "
            <<depth
            <<" score cp "
            <<score
            <<" nodes "
            <<nodes
            <<" pv "
            <<moveToUci(best)
            <<"\n";

        if(
            abs(score)>=
            MATE-100
        )
            break;

        if(
            chrono::steady_clock::now()>=
            stopAt
        )
            break;
    }

    if(best.from==best.to) {

        vector<Move> moves=
            legalMoves(false);

        if(!moves.empty())
            best=moves[0];
    }

    return best;
}

static void applyPosition(
    const string& line
) {

    stringstream ss(line);

    string token;

    ss>>token;

    ss>>token;

    if(token=="startpos") {

        setStart();

    } else if(token=="fen") {

        string fen;
        string x;

        for(int i=0;i<6;i++) {

            ss>>x;

            fen+=x;

            if(i<5)
                fen+=' ';
        }

        setFEN(fen);
    }

    if(ss>>token &&
       token=="moves") {

        while(ss>>token) {

            Move m=
                parseUci(token);

            Undo u;

            if(!makeMove(m,u))
                break;
        }
    }
}

int main() {

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    initZob();
    setStart();

    string line;

    while(getline(cin,line)) {

        if(line=="uci") {

            cout
                <<"id name SimpleLogics\n";

            cout
                <<"id author Danny\n";

            cout
                <<"option name Hash type spin "
                <<"default 64 min 1 max 1024\n";

            cout
                <<"option name Threads type spin "
                <<"default 1 min 1 max 1\n";

            cout<<"uciok\n";
        }

        else if(line=="isready") {

            cout<<"readyok\n";
        }

        else if(
            line.rfind(
                "setoption",0
            )==0
        ) {

            // Reserved for future options.
        }

        else if(
            line.rfind(
                "ucinewgame",0
            )==0
        ) {

            fill(
                tt.begin(),
                tt.end(),
                TTEntry{}
            );

            setStart();
        }

        else if(
            line.rfind(
                "position",0
            )==0
        ) {

            applyPosition(line);
        }

        else if(
            line.rfind(
                "go",0
            )==0
        ) {

            stringstream ss(line);

            string x;

            ss>>x;

            int movetime=1000;

            int wtime=-1;
            int btime=-1;

            int winc=0;
            int binc=0;

            int movesToGo=30;

            while(ss>>x) {

                if(x=="movetime")
                    ss>>movetime;

                else if(x=="wtime")
                    ss>>wtime;

                else if(x=="btime")
                    ss>>btime;

                else if(x=="winc")
                    ss>>winc;

                else if(x=="binc")
                    ss>>binc;

                else if(x=="movestogo")
                    ss>>movesToGo;

                else if(x=="depth") {

                    int d;

                    ss>>d;

                    movetime=
                        max(
                            100,
                            d*150
                        );
                }
            }

            if(
                movetime==1000 &&
                (wtime>=0 ||
                 btime>=0)
            ) {

                int time=
                    side==W
                    ? wtime
                    : btime;

                int inc=
                    side==W
                    ? winc
                    : binc;

                movetime=
                    max(
                        50,
                        time/
                        max(
                            5,
                            movesToGo
                        )
                        +
                        inc/2
                    );
            }

            Move best=
                think(movetime);

            cout
                <<"bestmove "
                <<moveToUci(best)
                <<"\n";
        }

        else if(line=="stop") {

            stopSearch=true;
        }

        else if(line=="quit") {

            break;
        }
    }

    return 0;
}
