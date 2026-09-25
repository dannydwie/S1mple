#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

/*
    SIMPLELOGICS 2
    ----------------
    Independent chess engine.

    Search:
      PVS
      Alpha-Beta
      Quiescence
      Transposition Table
      Null Move Pruning
      Late Move Reduction
      Futility Pruning
      Aspiration Windows
      Killer Moves
      History Heuristic
      MVV-LVA
      Check extensions

    Evaluation:
      Material
      Piece-square tables
      Bishop pair
      Mobility
      Pawn structure
      Passed pawns
      Rooks on open files
      King safety
*/

static const int INF  = 32000;
static const int MATE = 30000;

enum { WHITE=0, BLACK=1 };
enum { EMPTY=0, PAWN=1, KNIGHT=2, BISHOP=3,
       ROOK=4, QUEEN=5, KING=6 };

enum {
    FLAG_NONE=0,
    FLAG_EP=1,
    FLAG_CASTLE=2,
    FLAG_DOUBLE=4
};

struct Move {
    int from=0;
    int to=0;
    int promo=0;
    int flags=0;
    int capture=0;
    int score=0;

    bool operator==(const Move& x) const {
        return from==x.from &&
               to==x.to &&
               promo==x.promo;
    }
};

struct Undo {
    int captured;
    int castle;
    int ep;
    int half;
    int full;
};
static void undoMove(
    const Move& m,
    const Undo& u
);
struct TTEntry {
    uint64_t key=0;
    int depth=-1;
    int score=0;
    uint8_t flag=0;
    Move best{};
};

enum {
    TT_EXACT=0,
    TT_ALPHA=1,
    TT_BETA=2
};

static int board[128];

static int side=WHITE;
static int castleRights=0;
static int epSquare=-1;
static int halfmove=0;
static int fullmove=1;

static uint64_t hashKey=0;

static uint64_t zob[2][7][64];
static uint64_t zobCastle[16];
static uint64_t zobEP[64];
static uint64_t zobSide;

static vector<TTEntry> TT(1<<20);

static Move killers[128][2];
static int history[2][64][64];
static Move counterMove[64][64];

static int nodes=0;

static atomic<bool> stopFlag(false);

static chrono::steady_clock::time_point stopTime;

static const int value[7]={
    0,100,320,335,500,950,0
};

static const int knightDir[8]={
    -33,-31,-18,-14,
     14,18,31,33
};

static const int bishopDir[4]={
    -17,-15,15,17
};

static const int rookDir[4]={
    -16,-1,1,16
};

static const int kingDir[8]={
    -17,-16,-15,-1,1,15,16,17
};

static uint64_t random64() {

    static uint64_t x=
        0x9e3779b97f4a7c15ULL;

    x^=x>>12;
    x^=x<<25;
    x^=x>>27;

    return
        x*2685821657736338717ULL;
}

static int sq64(int s) {
    return
        (s&7)+
        ((s>>4)<<3);
}

static int fileOf(int s) {
    return s&7;
}

static int rankOf(int s) {
    return s>>4;
}

static int pieceType(int p) {
    return p>0?p:-p;
}

static int pieceColor(int p) {
    return p>0?WHITE:BLACK;
}

static void initZobrist() {

    for(int c=0;c<2;c++)
        for(int p=1;p<=6;p++)
            for(int s=0;s<64;s++)
                zob[c][p][s]=random64();

    for(int i=0;i<16;i++)
        zobCastle[i]=random64();

    for(int i=0;i<64;i++)
        zobEP[i]=random64();

    zobSide=random64();
}

static void hashPosition() {

    hashKey=0;

    for(int s=0;s<128;s++) {

        if(s&8)
            continue;

        int p=board[s];

        if(!p)
            continue;

        hashKey^=
            zob
            [pieceColor(p)]
            [pieceType(p)]
            [sq64(s)];
    }

    hashKey^=
        zobCastle[castleRights];

    if(epSquare>=0)
        hashKey^=
            zobEP[sq64(epSquare)];

    if(side==BLACK)
        hashKey^=zobSide;
}

static void clearBoard() {

    memset(board,0,sizeof(board));

    side=WHITE;
    castleRights=0;
    epSquare=-1;
    halfmove=0;
    fullmove=1;
}

static int parseSquare(
    const string& s
) {

    if(s.size()!=2)
        return -1;

    int f=s[0]-'a';
    int r=s[1]-'1';

    if(f<0||f>7||r<0||r>7)
        return -1;

    return r*16+f;
}

static string squareName(int s) {

    string x="a1";

    x[0]='a'+fileOf(s);
    x[1]='1'+rankOf(s);

    return x;
}

static void startPosition() {

    clearBoard();

    const string back=
        "RNBQKBNR";

    for(int f=0;f<8;f++) {

        board[f]=
            back[f]-'A'+1;

        board[16+f]=PAWN;

        board[96+f]=-PAWN;

        board[112+f]=
            -(back[f]-'A'+1);
    }

    castleRights=15;

    hashPosition();
}

static void setFEN(
    const string& fen
) {

    clearBoard();

    stringstream ss(fen);

    string placement;
    string stm;
    string cr;
    string ep;

    ss>>
        placement>>
        stm>>
        cr>>
        ep>>
        halfmove>>
        fullmove;

    int sq=112;

    for(char c:placement) {

        if(c=='/') {
            sq-=24;
            continue;
        }

        if(c>='1'&&c<='8') {
            sq+=c-'0';
            continue;
        }

        int p=0;

        switch(toupper(c)) {
            case 'P':p=PAWN;break;
            case 'N':p=KNIGHT;break;
            case 'B':p=BISHOP;break;
            case 'R':p=ROOK;break;
            case 'Q':p=QUEEN;break;
            case 'K':p=KING;break;
        }

        board[sq++]=
            isupper((unsigned char)c)
            ? p
            : -p;
    }

    side=
        stm=="b"
        ? BLACK
        : WHITE;

    castleRights=0;

    if(cr.find('K')!=string::npos)
        castleRights|=1;

    if(cr.find('Q')!=string::npos)
        castleRights|=2;

    if(cr.find('k')!=string::npos)
        castleRights|=4;

    if(cr.find('q')!=string::npos)
        castleRights|=8;

    epSquare=
        ep=="-"
        ? -1
        : parseSquare(ep);

    hashPosition();
}

static int kingSquare(int color) {

    int wanted=
        color==WHITE
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

static bool attacked(
    int sq,
    int by
) {

    int pawn=
        by==WHITE
        ? PAWN
        : -PAWN;

    int f=fileOf(sq);

    if(by==WHITE) {

        if(f>0 &&
           sq>=16 &&
           board[sq-17]==pawn)
            return true;

        if(f<7 &&
           sq>=16 &&
           board[sq-15]==pawn)
            return true;

    } else {

        if(f>0 &&
           sq<112 &&
           board[sq+15]==pawn)
            return true;

        if(f<7 &&
           sq<112 &&
           board[sq+17]==pawn)
            return true;
    }

    int knight=
        by==WHITE
        ? KNIGHT
        : -KNIGHT;

    for(int d:knightDir) {

        int x=sq+d;

        if(!(x&8) &&
           board[x]==knight)
            return true;
    }

    int bishop=
        by==WHITE
        ? BISHOP
        : -BISHOP;

    int queen=
        by==WHITE
        ? QUEEN
        : -QUEEN;

    for(int d:bishopDir) {

        for(
            int x=sq+d;
            !(x&8);
            x+=d
        ) {

            if(board[x]) {

                if(
                    board[x]==bishop ||
                    board[x]==queen
                )
                    return true;

                break;
            }
        }
    }

    int rook=
        by==WHITE
        ? ROOK
        : -ROOK;

    for(int d:rookDir) {

        for(
            int x=sq+d;
            !(x&8);
            x+=d
        ) {

            if(board[x]) {

                if(
                    board[x]==rook ||
                    board[x]==queen
                )
                    return true;

                break;
            }
        }
    }

    int king=
        by==WHITE
        ? KING
        : -KING;

    for(int d:kingDir) {

        int x=sq+d;

        if(!(x&8) &&
           board[x]==king)
            return true;
    }

    return false;
}

static bool inCheck(int color) {

    int k=kingSquare(color);

    return
        k>=0 &&
        attacked(k,color^1);
}

static void addMove(
    vector<Move>& list,
    int from,
    int to,
    int flags=0,
    int promo=0
) {

    Move m;

    m.from=from;
    m.to=to;
    m.flags=flags;
    m.promo=promo;
    m.capture=board[to];

    list.push_back(m);
}

static void generatePseudo(
    vector<Move>& list,
    bool capturesOnly=false
) {

    list.clear();

    for(int s=0;s<128;s++) {

        if(s&8)
            continue;

        int p=board[s];

        if(!p ||
           pieceColor(p)!=side)
            continue;

        int type=pieceType(p);

        if(type==PAWN) {

            int dir=
                side==WHITE
                ? 16
                : -16;

            int startRank=
                side==WHITE
                ? 1
                : 6;

            int promoRank=
                side==WHITE
                ? 7
                : 0;

            int one=s+dir;

            if(
                !capturesOnly &&
                !(one&8) &&
                !board[one]
            ) {

                if(rankOf(one)==promoRank) {

                    addMove(
                        list,s,one,0,QUEEN
                    );

                    addMove(
                        list,s,one,0,ROOK
                    );

                    addMove(
                        list,s,one,0,BISHOP
                    );

                    addMove(
                        list,s,one,0,KNIGHT
                    );

                } else {

                    addMove(
                        list,s,one
                    );

                    int two=s+dir*2;

                    if(
                        rankOf(s)==startRank &&
                        !board[two]
                    )
                        addMove(
                            list,
                            s,
                            two,
                            FLAG_DOUBLE
                        );
                }
            }

            for(int df:{-1,1}) {

                int x=s+dir+df;

                if(x&8)
                    continue;

                if(
                    board[x] &&
                    pieceColor(board[x])!=side
                ) {

                    if(rankOf(x)==promoRank) {

                        addMove(
                            list,s,x,0,QUEEN
                        );

                        addMove(
                            list,s,x,0,ROOK
                        );

                        addMove(
                            list,s,x,0,BISHOP
                        );

                        addMove(
                            list,s,x,0,KNIGHT
                        );

                    } else {

                        addMove(
                            list,s,x
                        );
                    }

                } else if(x==epSquare) {

                    addMove(
                        list,
                        s,
                        x,
                        FLAG_EP
                    );
                }
            }

        } else if(type==KNIGHT) {

            for(int d:knightDir) {

                int x=s+d;

                if(x&8)
                    continue;

                if(
                    !board[x] ||
                    pieceColor(board[x])!=side
                ) {

                    if(
                        !capturesOnly ||
                        board[x]
                    )
                        addMove(
                            list,s,x
                        );
                }
            }

        } else if(
            type==BISHOP ||
            type==ROOK ||
            type==QUEEN
        ) {

            int dirs[8];
            int n=0;

            if(
                type==BISHOP ||
                type==QUEEN
            ) {

                for(int d:bishopDir)
                    dirs[n++]=d;
            }

            if(
                type==ROOK ||
                type==QUEEN
            ) {

                for(int d:rookDir)
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
                            addMove(
                                list,s,x
                            );

                    } else {

                        if(
                            pieceColor(board[x])
                            !=side
                        )
                            addMove(
                                list,s,x
                            );

                        break;
                    }
                }
            }

        } else {

            for(int d:kingDir) {

                int x=s+d;

                if(x&8)
                    continue;

                if(
                    !board[x] ||
                    pieceColor(board[x])!=side
                ) {

                    if(
                        !capturesOnly ||
                        board[x]
                    )
                        addMove(
                            list,s,x
                        );
                }
            }

            if(
                !capturesOnly &&
                !inCheck(side)
            ) {

                if(
                    side==WHITE &&
                    (castleRights&1) &&
                    !board[5] &&
                    !board[6] &&
                    !attacked(5,BLACK) &&
                    !attacked(6,BLACK)
                )
                    addMove(
                        list,4,6,
                        FLAG_CASTLE
                    );

                if(
                    side==WHITE &&
                    (castleRights&2) &&
                    !board[3] &&
                    !board[2] &&
                    !board[1] &&
                    !attacked(3,BLACK) &&
                    !attacked(2,BLACK)
                )
                    addMove(
                        list,4,2,
                        FLAG_CASTLE
                    );

                if(
                    side==BLACK &&
                    (castleRights&4) &&
                    !board[117] &&
                    !board[118] &&
                    !attacked(117,WHITE) &&
                    !attacked(118,WHITE)
                )
                    addMove(
                        list,116,118,
                        FLAG_CASTLE
                    );

                if(
                    side==BLACK &&
                    (castleRights&8) &&
                    !board[115] &&
                    !board[114] &&
                    !board[113] &&
                    !attacked(115,WHITE) &&
                    !attacked(114,WHITE)
                )
                    addMove(
                        list,116,114,
                        FLAG_CASTLE
                    );
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
    u.ep=epSquare;
    u.half=halfmove;
    u.full=fullmove;

    int p=board[m.from];
    int us=side;

    board[m.from]=0;
    board[m.to]=p;

    if(m.flags&FLAG_EP)
        board[
            m.to+(us==WHITE?-16:16)
        ]=0;

    if(m.promo)
        board[m.to]=
            us==WHITE
            ? m.promo
            : -m.promo;

    if(m.flags&FLAG_CASTLE) {

        if(m.to==6) {
            board[5]=board[7];
            board[7]=0;
        }

        if(m.to==2) {
            board[3]=board[0];
            board[0]=0;
        }

        if(m.to==118) {
            board[117]=board[119];
            board[119]=0;
        }

        if(m.to==114) {
            board[115]=board[112];
            board[112]=0;
        }
    }

    if(pieceType(p)==KING) {

        if(us==WHITE)
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

    epSquare=-1;

    if(m.flags&FLAG_DOUBLE)
        epSquare=
            m.from+
            (us==WHITE?16:-16);

    if(
        pieceType(p)==PAWN ||
        u.captured ||
        (m.flags&FLAG_EP)
    )
        halfmove=0;
    else
        halfmove++;

    if(us==BLACK)
        fullmove++;

    side^=1;

    hashPosition();

    if(inCheck(us)) {

        undoMove(m,u);

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
    epSquare=u.ep;
    halfmove=u.half;
    fullmove=u.full;

    int p=board[m.to];

    if(m.promo)
        p=
            side==WHITE
            ? PAWN
            : -PAWN;

    board[m.from]=p;
    board[m.to]=u.captured;

    if(m.flags&FLAG_EP)
        board[
            m.to+(side==WHITE?-16:16)
        ]=
            side==WHITE
            ? -PAWN
            : PAWN;

    if(m.flags&FLAG_CASTLE) {

        if(m.to==6) {
            board[7]=board[5];
            board[5]=0;
        }

        if(m.to==2) {
            board[0]=board[3];
            board[3]=0;
        }

        if(m.to==118) {
            board[119]=board[117];
            board[117]=0;
        }

        if(m.to==114) {
            board[112]=board[115];
            board[115]=0;
        }
    }

    hashPosition();
}

static vector<Move> legalMoves(
    bool capturesOnly=false
) {

    vector<Move> pseudo;
    vector<Move> legal;

    generatePseudo(
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
   EVALUATION
   ========================= */

static const int pawnPSQT[64]={
     0,  5,  5,  0,  0,  5,  5,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  5, 10, 20, 20, 10,  5,  0,
     0,  5, 15, 25, 25, 15,  5,  0,
     5, 10, 20, 30, 30, 20, 10,  5,
    10, 15, 25, 35, 35, 25, 15, 10,
    40, 40, 40, 40, 40, 40, 40, 40,
     0,  0,  0,  0,  0,  0,  0,  0
};

static const int knightPSQT[64]={
   -40,-30,-20,-20,-20,-20,-30,-40,
   -30,-10,  0,  5,  5,  0,-10,-30,
   -20,  5, 10, 15, 15, 10,  5,-20,
   -20,  0, 15, 20, 20, 15,  0,-20,
   -20,  5, 15, 20, 20, 15,  5,-20,
   -20,  0, 10, 15, 15, 10,  0,-20,
   -30,-10,  0,  0,  0,  0,-10,-30,
   -40,-30,-20,-20,-20,-20,-30,-40
};

static const int bishopPSQT[64]={
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5, 10, 15, 15, 10,  5,-10,
   -10,  0, 10, 15, 15, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};

static const int kingPSQT[64]={
   -50,-40,-40,-50,-50,-40,-40,-50,
   -40,-30,-30,-40,-40,-30,-30,-40,
   -30,-20,-20,-30,-30,-20,-20,-30,
   -20,-10,-10,-20,-20,-10,-10,-20,
     0,  0,  0,  0,  0,  0,  0,  0,
    20, 20, 10,  0,  0, 10, 20, 20,
    30, 30, 20, 10, 10, 20, 30, 30,
    30, 40, 20,  0,  0, 20, 40, 30
};

static int mirror64(int s) {

    return
        (7-rankOf(s))*8+
        fileOf(s);
}

static int evaluate() {

    int score=0;

    int bishops[2]={0,0};

    int pawns[2][8]={};

    int kingSq[2]={-1,-1};

    for(int s=0;s<128;s++) {

        if(s&8)
            continue;

        int p=board[s];

        if(!p)
            continue;

        int c=pieceColor(p);
        int t=pieceType(p);

        int sq=
            c==WHITE
            ? sq64(s)
            : mirror64(s);

        int v=value[t];

        if(t==PAWN)
            v+=pawnPSQT[sq];

        if(t==KNIGHT)
            v+=knightPSQT[sq];

        if(t==BISHOP)
            v+=bishopPSQT[sq];

        if(t==KING)
            v+=kingPSQT[sq];

        if(t==BISHOP)
            bishops[c]++;

        if(t==PAWN)
            pawns[c][fileOf(s)]++;

        if(t==KING)
            kingSq[c]=s;

        score+=
            c==WHITE
            ? v
            : -v;
    }

    if(bishops[WHITE]>=2)
        score+=35;

    if(bishops[BLACK]>=2)
        score-=35;

    /*
       Doubled pawns.
    */

    for(int c=0;c<2;c++) {

        for(int f=0;f<8;f++) {

            if(pawns[c][f]>1) {

                int penalty=
                    12*(pawns[c][f]-1);

                score+=
                    c==WHITE
                    ? -penalty
                    : penalty;
            }
        }
    }

    /*
       Open/semi-open rook files.
    */

    for(int s=0;s<128;s++) {

        if(s&8)
            continue;

        int p=board[s];

        if(!p ||
           pieceType(p)!=ROOK)
            continue;

        int f=fileOf(s);
        int c=pieceColor(p);

        bool ownPawn=false;
        bool enemyPawn=false;

        for(int r=0;r<8;r++) {

            int x=r*16+f;

            if(
                pieceType(board[x])==
                PAWN
            ) {

                if(
                    pieceColor(board[x])==c
                )
                    ownPawn=true;
                else
                    enemyPawn=true;
            }
        }

        if(!ownPawn) {

            int bonus=
                enemyPawn
                ? 12
                : 22;

            score+=
                c==WHITE
                ? bonus
                : -bonus;
        }
    }

    /*
       Passed pawns.
    */

    for(int s=0;s<128;s++) {

        if(s&8)
            continue;

        if(pieceType(board[s])!=PAWN)
            continue;

        int c=pieceColor(board[s]);
        int f=fileOf(s);
        bool passed=true;

        for(int df=-1;df<=1;df++) {

            int ff=f+df;

            if(ff<0||ff>7)
                continue;

            for(int r=0;r<8;r++) {

                int x=r*16+ff;

                if(
                    pieceType(board[x])==PAWN &&
                    pieceColor(board[x])!=c
                ) {

                    if(
                        c==WHITE
                        ? r>rankOf(s)
                        : r<rankOf(s)
                    )
                        passed=false;
                }
            }
        }

        if(passed) {

            int advance=
                c==WHITE
                ? rankOf(s)
                : 7-rankOf(s);

            int bonus=
                10+advance*8;

            score+=
                c==WHITE
                ? bonus
                : -bonus;
        }
    }

    /*
       Mobility.
    */

    int saveSide=side;

    vector<Move> a;
    vector<Move> b;

    generatePseudo(a,false);

    side^=1;

    generatePseudo(b,false);

    side=saveSide;

    int mobility=
        (int)a.size()-
        (int)b.size();

    score+=
        saveSide==WHITE
        ? mobility*2
        : -mobility*2;

    /*
       King safety.
    */

    if(kingSq[WHITE]>=0) {

        int f=fileOf(kingSq[WHITE]);

        int attacks=0;

        for(int df=-1;df<=1;df++) {

            int ff=f+df;

            if(ff<0||ff>7)
                continue;

            for(int r=0;r<8;r++) {

                int x=r*16+ff;

                if(
                    board[x] &&
                    pieceColor(board[x])==BLACK
                )
                    attacks++;
            }
        }

        score-=attacks*3;
    }

    if(kingSq[BLACK]>=0) {

        int f=fileOf(kingSq[BLACK]);

        int attacks=0;

        for(int df=-1;df<=1;df++) {

            int ff=f+df;

            if(ff<0||ff>7)
                continue;

            for(int r=0;r<8;r++) {

                int x=r*16+ff;

                if(
                    board[x] &&
                    pieceColor(board[x])==WHITE
                )
                    attacks++;
            }
        }

        score+=attacks*3;
    }

    return
        side==WHITE
        ? score
        : -score;
}

/* =========================
   MOVE ORDERING
   ========================= */

static int moveValue(
    const Move& m
) {

    if(m.capture) {

        return
            100000+
            10*value[
                pieceType(m.capture)
            ]-
            value[
                pieceType(
                    board[m.from]
                )
            ];
    }

    if(m.promo)
        return
            90000+
            value[m.promo];

    if(m==killers[0][0])
        return 80000;

    if(m==killers[0][1])
        return 79000;

    int c=side;

    return
        history[c]
        [sq64(m.from)]
        [sq64(m.to)];
}

static void orderMoves(
    vector<Move>& moves,
    const Move* hashMove=nullptr
) {

    for(auto &m:moves) {

        m.score=moveValue(m);

        if(
            hashMove &&
            m==*hashMove
        )
            m.score+=1000000;
    }

    sort(
        moves.begin(),
        moves.end(),
        [](const Move&a,const Move&b) {
            return a.score>b.score;
        }
    );
}

/* =========================
   QUIESCENCE
   ========================= */

static int quiescence(
    int alpha,
    int beta,
    int ply
) {

    nodes++;

    if(
        stopFlag ||
        (
            (nodes&2047)==0 &&
            chrono::steady_clock::now()>=stopTime
        )
    ) {

        stopFlag=true;
        return 0;
    }

    bool check=inCheck(side);

    int stand=evaluate();

    if(!check) {

        if(stand>=beta)
            return beta;

        if(stand>alpha)
            alpha=stand;
    }

    vector<Move> moves=
        legalMoves(!check);

    orderMoves(moves);

    for(auto &m:moves) {

        Undo u;

        if(!makeMove(m,u))
            continue;

        int score=
            -quiescence(
                -beta,
                -alpha,
                ply+1
            );

        undoMove(m,u);

        if(stopFlag)
            return 0;

        if(score>=beta)
            return beta;

        if(score>alpha)
            alpha=score;
    }

    return alpha;
}

/* =========================
   SEARCH
   ========================= */

static int search(
    int depth,
    int alpha,
    int beta,
    int ply,
    bool allowNull
) {

    if(
        stopFlag ||
        (
            (nodes&2047)==0 &&
            chrono::steady_clock::now()>=stopTime
        )
    ) {

        stopFlag=true;
        return 0;
    }

    nodes++;

    bool check=inCheck(side);

    if(depth<=0)
        return quiescence(
            alpha,
            beta,
            ply
        );

    size_t index=
        hashKey&
        (TT.size()-1);

    TTEntry &entry=TT[index];

    Move hashMove=entry.best;

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

    /*
       Null move pruning.
    */

    if(
        allowNull &&
        !check &&
        depth>=3 &&
        ply>0
    ) {

        bool hasBigPiece=false;

        for(int s=0;s<128;s++) {

            if(s&8)
                continue;

            if(
                board[s] &&
                pieceColor(board[s])==side &&
                pieceType(board[s])>=KNIGHT
            ) {

                hasBigPiece=true;
                break;
            }
        }

        if(hasBigPiece) {

            int oldEP=epSquare;

            epSquare=-1;

            side^=1;

            hashPosition();

            int reduction=
                depth>=6
                ? 3
                : 2;

            int score=
                -search(
                    depth-reduction-1,
                    -beta,
                    -beta+1,
                    ply+1,
                    false
                );

            side^=1;

            epSquare=oldEP;

            hashPosition();

            if(stopFlag)
                return 0;

            if(score>=beta)
                return beta;
        }
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
        ? &hashMove
        : nullptr
    );

    int originalAlpha=alpha;

    int bestScore=-INF;

    Move bestMove=moves[0];

    int moveNumber=0;

    for(auto &m:moves) {

        Undo u;

        if(!makeMove(m,u))
            continue;

        int newDepth=
            depth-1;

        /*
           Check extension.
        */

        bool givesCheck=
            inCheck(side);

        if(givesCheck)
            newDepth++;

        /*
           Late Move Reduction.
        */

        if(
            moveNumber>=4 &&
            depth>=3 &&
            !check &&
            !m.capture &&
            !m.promo &&
            !givesCheck
        ) {

            int reduction=1;

            if(
                moveNumber>=8 &&
                depth>=6
            )
                reduction=2;

            newDepth=
                max(
                    1,
                    newDepth-reduction
                );
        }

        int score;

        /*
           Principal variation search.
        */

        if(moveNumber==0) {

            score=
                -search(
                    newDepth,
                    -beta,
                    -alpha,
                    ply+1,
                    true
                );

        } else {

            score=
                -search(
                    newDepth,
                    -alpha-1,
                    -alpha,
                    ply+1,
                    true
                );

            if(
                score>alpha &&
                score<beta
            ) {

                score=
                    -search(
                        newDepth,
                        -beta,
                        -alpha,
                        ply+1,
                        true
                    );
            }
        }

        undoMove(m,u);

        if(stopFlag)
            return 0;

        if(score>bestScore) {

            bestScore=score;
            bestMove=m;
        }

        if(score>alpha)
            alpha=score;

        if(alpha>=beta) {

            /*
               Killer move.
            */

            if(!m.capture) {

                killers[ply][1]=
                    killers[ply][0];

                killers[ply][0]=m;

                int c=side;

                history[c]
                [sq64(m.from)]
                [sq64(m.to)]
                +=depth*depth;

                if(
                    history[c]
                    [sq64(m.from)]
                    [sq64(m.to)]
                    >100000
                )
                    history[c]
                    [sq64(m.from)]
                    [sq64(m.to)]
                    =100000;
            }

            break;
        }

        moveNumber++;
    }

    entry.key=hashKey;
    entry.depth=depth;
    entry.score=bestScore;
    entry.best=bestMove;

    if(bestScore<=originalAlpha)
        entry.flag=TT_ALPHA;

    else if(bestScore>=beta)
        entry.flag=TT_BETA;

    else
        entry.flag=TT_EXACT;

    return bestScore;
}

/* =========================
   ITERATIVE DEEPENING
   ========================= */

static string uciMove(
    const Move& m
) {

    string s=
        squareName(m.from)+
        squareName(m.to);

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

static Move findBestMove(
    int timeMs
) {

    stopFlag=false;
    nodes=0;

    stopTime=
        chrono::steady_clock::now()+
        chrono::milliseconds(
            max(100,timeMs)
        );

    Move best{};

    int previousScore=0;

    for(int depth=1;depth<=64;depth++) {

        int alpha=-INF;
        int beta=INF;

        /*
           Aspiration window.
        */

        if(depth>=5) {

            alpha=
                previousScore-50;

            beta=
                previousScore+50;
        }

        int score=
            search(
                depth,
                alpha,
                beta,
                0,
                true
            );

        if(stopFlag)
            break;

        /*
           Re-search after aspiration fail.
        */

        if(
            depth>=5 &&
            (score<=alpha ||
             score>=beta)
        ) {

            score=
                search(
                    depth,
                    -INF,
                    INF,
                    0,
                    true
                );

            if(stopFlag)
                break;
        }

        previousScore=score;

        TTEntry &e=
            TT[
                hashKey&
                (TT.size()-1)
            ];

        if(e.key==hashKey)
            best=e.best;

        cout
            <<"info depth "
            <<depth
            <<" score cp "
            <<score
            <<" nodes "
            <<nodes
            <<" pv "
            <<uciMove(best)
            <<"\n";

        if(
            abs(score)>=
            MATE-100
        )
            break;

        if(
            chrono::steady_clock::now()>=
            stopTime
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

static Move parseMove(
    const string& s
) {

    Move result{};

    if(s.size()<4)
        return result;

    result.from=
        parseSquare(
            s.substr(0,2)
        );

    result.to=
        parseSquare(
            s.substr(2,2)
        );

    if(s.size()>4) {

        char c=
            tolower(
                (unsigned char)s[4]
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

static void handlePosition(
    const string& line
) {

    stringstream ss(line);

    string token;

    ss>>token;

    ss>>token;

    if(token=="startpos") {

        startPosition();

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
                parseMove(token);

            Undo u;

            if(!makeMove(m,u))
                break;
        }
    }
}

int main() {

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    initZobrist();

    startPosition();

    string line;

    while(getline(cin,line)) {

        if(line=="uci") {

            cout
                <<"id name SimpleLogics 2\n";

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
                "ucinewgame",0
            )==0
        ) {

            fill(
                TT.begin(),
                TT.end(),
                TTEntry{}
            );

            memset(
                history,
                0,
                sizeof(history)
            );

            memset(
                killers,
                0,
                sizeof(killers)
            );

            startPosition();
        }

        else if(
            line.rfind(
                "position",0
            )==0
        ) {

            handlePosition(line);
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
            }

            if(
                movetime==1000 &&
                (wtime>=0 ||
                 btime>=0)
            ) {

                int t=
                    side==WHITE
                    ? wtime
                    : btime;

                int inc=
                    side==WHITE
                    ? winc
                    : binc;

                movetime=
                    max(
                        100,
                        t/
                        max(
                            8,
                            movesToGo
                        )
                        +
                        inc/2
                    );
            }

            Move best=
                findBestMove(
                    movetime
                );

            cout
                <<"bestmove "
                <<uciMove(best)
                <<"\n";
        }

        else if(line=="stop") {

            stopFlag=true;
        }

        else if(line=="quit") {

            break;
        }
    }

    return 0;
}
