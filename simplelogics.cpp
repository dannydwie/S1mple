// SimpleLogics NNUE Engine
// Independent chess engine.
// NNUE format: SLNNUE2
//
// UCI:
//   uci
//   isready
//   position startpos
//   position fen ...
//   go depth N
//   go movetime N
//   stop
//   quit

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

static constexpr int EMPTY = 0;
static constexpr int WP=1, WN=2, WB=3, WR=4, WQ=5, WK=6;
static constexpr int BP=7, BN=8, BB=9, BR=10, BQ=11, BK=12;

static constexpr int WHITE=0;
static constexpr int BLACK=1;

static constexpr int INF=32000;
static constexpr int MATE=30000;
static constexpr int MAX_PLY=128;

struct Move {
    int from=0;
    int to=0;
    int promotion=0;
    int flags=0;
    int score=0;
};

struct Undo {
    int captured=EMPTY;
    int castling=0;
    int ep=-1;
    int halfmove=0;
};

struct Position {
    array<int,64> board{};
    int side=WHITE;
    int castling=15;
    int ep=-1;
    int halfmove=0;
    int fullmove=1;
};

static Position pos;

static uint64_t zobrist[13][64];
static uint64_t zobSide;
static uint64_t zobCastle[16];
static uint64_t zobEP[65];

static atomic<bool> stopSearch(false);

static int maxDepth=64;
static long long stopTime=0;

static int historyTable[2][64][64];
static Move killers[MAX_PLY][2];

struct TTEntry {
    uint64_t key=0;
    int depth=-1;
    int score=0;
    int flag=0;
    Move best{};
};

static constexpr size_t TT_SIZE=1<<20;
static vector<TTEntry> TT(TT_SIZE);

static uint64_t rng64() {
    static uint64_t x=0x9e3779b97f4a7c15ULL;

    x^=x>>12;
    x^=x<<25;
    x^=x>>27;

    return x*2685821657736338717ULL;
}

static void initZobrist() {

    for(int p=0;p<13;p++)
        for(int s=0;s<64;s++)
            zobrist[p][s]=rng64();

    zobSide=rng64();

    for(auto &x:zobCastle)
        x=rng64();

    for(auto &x:zobEP)
        x=rng64();
}

static uint64_t hashPosition() {

    uint64_t h=0;

    for(int s=0;s<64;s++) {
        int p=pos.board[s];

        if(p)
            h^=zobrist[p][s];
    }

    if(pos.side==BLACK)
        h^=zobSide;

    h^=zobCastle[pos.castling];

    if(pos.ep>=0)
        h^=zobEP[pos.ep];

    return h;
}

static inline int fileOf(int s) {
    return s&7;
}

static inline int rankOf(int s) {
    return s>>3;
}

static inline bool whitePiece(int p) {
    return p>=WP && p<=WK;
}

static inline bool blackPiece(int p) {
    return p>=BP && p<=BK;
}

static inline bool ownPiece(int p,int side) {
    return side==WHITE ? whitePiece(p) : blackPiece(p);
}

static inline bool enemyPiece(int p,int side) {
    return side==WHITE ? blackPiece(p) : whitePiece(p);
}

static int pieceValue(int p) {

    switch(p) {
        case WP:
        case BP: return 100;

        case WN:
        case BN: return 320;

        case WB:
        case BB: return 330;

        case WR:
        case BR: return 500;

        case WQ:
        case BQ: return 900;

        case WK:
        case BK: return 20000;
    }

    return 0;
}

static bool squareAttacked(int sq,int bySide) {

    int r=rankOf(sq);
    int f=fileOf(sq);

    // Pawns
    if(bySide==WHITE) {

        if(r>0 && f>0 && pos.board[sq-9]==WP)
            return true;

        if(r>0 && f<7 && pos.board[sq-7]==WP)
            return true;

    } else {

        if(r<7 && f>0 && pos.board[sq+7]==BP)
            return true;

        if(r<7 && f<7 && pos.board[sq+9]==BP)
            return true;
    }

    // Knights
    static const int knr[8]={
        -2,-2,-1,-1,1,1,2,2
    };

    static const int knf[8]={
        -1,1,-2,2,-2,2,-1,1
    };

    for(int i=0;i<8;i++) {

        int rr=r+knr[i];
        int ff=f+knf[i];

        if(rr<0||rr>7||ff<0||ff>7)
            continue;

        int p=pos.board[rr*8+ff];

        if(p==(bySide==WHITE?WN:BN))
            return true;
    }

    // Bishops / queens
    static const int drB[4]={
        1,1,-1,-1
    };

    static const int dfB[4]={
        1,-1,1,-1
    };

    for(int d=0;d<4;d++) {

        int rr=r+drB[d];
        int ff=f+dfB[d];

        while(rr>=0&&rr<8&&ff>=0&&ff<8) {

            int p=pos.board[rr*8+ff];

            if(p) {

                if(p==(bySide==WHITE?WB:BB) ||
                   p==(bySide==WHITE?WQ:BQ))
                    return true;

                break;
            }

            rr+=drB[d];
            ff+=dfB[d];
        }
    }

    // Rooks / queens
    static const int drR[4]={
        1,-1,0,0
    };

    static const int dfR[4]={
        0,0,1,-1
    };

    for(int d=0;d<4;d++) {

        int rr=r+drR[d];
        int ff=f+dfR[d];

        while(rr>=0&&rr<8&&ff>=0&&ff<8) {

            int p=pos.board[rr*8+ff];

            if(p) {

                if(p==(bySide==WHITE?WR:BR) ||
                   p==(bySide==WHITE?WQ:BQ))
                    return true;

                break;
            }

            rr+=drR[d];
            ff+=dfR[d];
        }
    }

    // King
    for(int rr=max(0,r-1);rr<=min(7,r+1);rr++) {

        for(int ff=max(0,f-1);ff<=min(7,f+1);ff++) {

            if(rr==r && ff==f)
                continue;

            if(pos.board[rr*8+ff]==(bySide==WHITE?WK:BK))
                return true;
        }
    }

    return false;
}

static bool inCheck(int side) {

    int king=(side==WHITE?WK:BK);

    for(int s=0;s<64;s++) {

        if(pos.board[s]==king)
            return squareAttacked(s,side^1);
    }

    return true;
}

static void addMove(
    vector<Move>& moves,
    int from,
    int to,
    int promotion=0,
    int flags=0
) {

    Move m;

    m.from=from;
    m.to=to;
    m.promotion=promotion;
    m.flags=flags;

    int captured=pos.board[to];

    if(flags&1)
        captured=(pos.side==WHITE?BP:WP);

    if(captured)
        m.score=100000+
            pieceValue(captured)*10-
            pieceValue(pos.board[from]);

    moves.push_back(m);
}

static void generatePseudo(vector<Move>& moves) {

    moves.clear();

    int side=pos.side;

    for(int sq=0;sq<64;sq++) {

        int p=pos.board[sq];

        if(!ownPiece(p,side))
            continue;

        int r=rankOf(sq);
        int f=fileOf(sq);

        // Pawn
        if(p==WP || p==BP) {

            int dir=(p==WP?1:-1);
            int start=(p==WP?1:6);
            int promoRank=(p==WP?7:0);

            int nr=r+dir;

            if(nr>=0&&nr<8) {

                int to=nr*8+f;

                if(pos.board[to]==EMPTY) {

                    if(nr==promoRank) {

                        addMove(moves,sq,to,WQ);
                        addMove(moves,sq,to,WR);
                        addMove(moves,sq,to,WB);
                        addMove(moves,sq,to,WN);

                    } else {

                        addMove(moves,sq,to);

                        if(r==start) {

                            int to2=(r+2*dir)*8+f;

                            if(pos.board[to2]==EMPTY)
                                addMove(moves,sq,to2,0,2);
                        }
                    }
                }
            }

            for(int df:{-1,1}) {

                int ff=f+df;
                int rr=r+dir;

                if(ff<0||ff>7||rr<0||rr>7)
                    continue;

                int to=rr*8+ff;

                if(enemyPiece(pos.board[to],side)) {

                    if(rr==promoRank) {

                        addMove(moves,sq,to,WQ);
                        addMove(moves,sq,to,WR);
                        addMove(moves,sq,to,WB);
                        addMove(moves,sq,to,WN);

                    } else {

                        addMove(moves,sq,to);
                    }
                }

                if(to==pos.ep)
                    addMove(moves,sq,to,0,1);
            }

            continue;
        }

        // Knight
        if(p==WN || p==BN) {

            static const int dr[8]={
                -2,-2,-1,-1,1,1,2,2
            };

            static const int df[8]={
                -1,1,-2,2,-2,2,-1,1
            };

            for(int i=0;i<8;i++) {

                int rr=r+dr[i];
                int ff=f+df[i];

                if(rr<0||rr>7||ff<0||ff>7)
                    continue;

                int to=rr*8+ff;

                if(!ownPiece(pos.board[to],side))
                    addMove(moves,sq,to);
            }

            continue;
        }

        // Bishop / rook / queen
        if(p==WB||p==BB||p==WR||p==BR||p==WQ||p==BQ) {

            static const int dr[8]={
                1,-1,0,0,1,1,-1,-1
            };

            static const int df[8]={
                0,0,1,-1,1,-1,1,-1
            };

            int begin=0;
            int end=8;

            if(p==WB||p==BB)
                begin=4;

            if(p==WR||p==BR)
                end=4;

            for(int d=begin;d<end;d++) {

                int rr=r+dr[d];
                int ff=f+df[d];

                while(rr>=0&&rr<8&&ff>=0&&ff<8) {

                    int to=rr*8+ff;

                    if(ownPiece(pos.board[to],side))
                        break;

                    addMove(moves,sq,to);

                    if(pos.board[to])
                        break;

                    rr+=dr[d];
                    ff+=df[d];
                }
            }

            continue;
        }

        // King
        if(p==WK || p==BK) {

            for(int rr=max(0,r-1);rr<=min(7,r+1);rr++) {

                for(int ff=max(0,f-1);ff<=min(7,f+1);ff++) {

                    if(rr==r&&ff==f)
                        continue;

                    int to=rr*8+ff;

                    if(!ownPiece(pos.board[to],side))
                        addMove(moves,sq,to);
                }
            }

            // Castling
            if(side==WHITE) {

                if((pos.castling&1) &&
                   pos.board[5]==EMPTY &&
                   pos.board[6]==EMPTY &&
                   !squareAttacked(4,BLACK) &&
                   !squareAttacked(5,BLACK) &&
                   !squareAttacked(6,BLACK))
                    addMove(moves,4,6,0,4);

                if((pos.castling&2) &&
                   pos.board[3]==EMPTY &&
                   pos.board[2]==EMPTY &&
                   pos.board[1]==EMPTY &&
                   !squareAttacked(4,BLACK) &&
                   !squareAttacked(3,BLACK) &&
                   !squareAttacked(2,BLACK))
                    addMove(moves,4,2,0,4);

            } else {

                if((pos.castling&4) &&
                   pos.board[61]==EMPTY &&
                   pos.board[62]==EMPTY &&
                   !squareAttacked(60,WHITE) &&
                   !squareAttacked(61,WHITE) &&
                   !squareAttacked(62,WHITE))
                    addMove(moves,60,62,0,4);

                if((pos.castling&8) &&
                   pos.board[59]==EMPTY &&
                   pos.board[58]==EMPTY &&
                   pos.board[57]==EMPTY &&
                   !squareAttacked(60,WHITE) &&
                   !squareAttacked(59,WHITE) &&
                   !squareAttacked(58,WHITE))
                    addMove(moves,60,58,0,4);
            }
        }
    }
}

static bool makeMove(const Move& m,Undo& u) {

    u.captured=pos.board[m.to];

    if(m.flags&1) {

        u.captured=(pos.side==WHITE?BP:WP);

        int csq=pos.side==WHITE?m.to-8:m.to+8;

        pos.board[csq]=EMPTY;
    }

    u.castling=pos.castling;
    u.ep=pos.ep;
    u.halfmove=pos.halfmove;

    int p=pos.board[m.from];

    pos.board[m.from]=EMPTY;
    pos.board[m.to]=m.promotion?(
        pos.side==WHITE?m.promotion:m.promotion+6
    ):p;

    if(p==WK) pos.castling&=~3;
    if(p==BK) pos.castling&=~12;

    if(m.from==0||m.to==0) pos.castling&=~2;
    if(m.from==7||m.to==7) pos.castling&=~1;
    if(m.from==56||m.to==56) pos.castling&=~8;
    if(m.from==63||m.to==63) pos.castling&=~4;

    if(p==WP||p==BP||u.captured)
        pos.halfmove=0;
    else
        pos.halfmove++;

    pos.ep=-1;

    if(m.flags&2)
        pos.ep=(pos.side==WHITE?m.from+8:m.from-8);

    if(m.flags&4) {

        if(m.to==6) {
            pos.board[5]=WR;
            pos.board[7]=EMPTY;
        }

        if(m.to==2) {
            pos.board[3]=WR;
            pos.board[0]=EMPTY;
        }

        if(m.to==62) {
            pos.board[61]=BR;
            pos.board[63]=EMPTY;
        }

        if(m.to==58) {
            pos.board[59]=BR;
            pos.board[56]=EMPTY;
        }
    }

    pos.side^=1;

    if(pos.side==WHITE)
        pos.fullmove++;

    if(inCheck(pos.side^1)) {

        pos.side^=1;

        pos.board[m.from]=p;
        pos.board[m.to]=u.captured;

        if(m.flags&1) {

            int csq=pos.side==WHITE?m.to-8:m.to+8;
            pos.board[csq]=(pos.side==WHITE?BP:WP);
        }

        if(m.flags&4) {

            if(m.to==6) {
                pos.board[7]=WR;
                pos.board[5]=EMPTY;
            }

            if(m.to==2) {
                pos.board[0]=WR;
                pos.board[3]=EMPTY;
            }

            if(m.to==62) {
                pos.board[63]=BR;
                pos.board[61]=EMPTY;
            }

            if(m.to==58) {
                pos.board[56]=BR;
                pos.board[59]=EMPTY;
            }
        }

        pos.castling=u.castling;
        pos.ep=u.ep;
        pos.halfmove=u.halfmove;

        return false;
    }

    return true;
}

static void undoMove(const Move& m,const Undo& u) {

    pos.side^=1;

    int p=pos.board[m.to];

    if(m.promotion)
        p=(pos.side==WHITE?WP:BP);

    pos.board[m.from]=p;
    pos.board[m.to]=u.captured;

    if(m.flags&1) {

        int csq=pos.side==WHITE?m.to-8:m.to+8;
        pos.board[csq]=(pos.side==WHITE?BP:WP);
    }

    if(m.flags&4) {

        if(m.to==6) {
            pos.board[7]=WR;
            pos.board[5]=EMPTY;
        }

        if(m.to==2) {
            pos.board[0]=WR;
            pos.board[3]=EMPTY;
        }

        if(m.to==62) {
            pos.board[63]=BR;
            pos.board[61]=EMPTY;
        }

        if(m.to==58) {
            pos.board[56]=BR;
            pos.board[59]=EMPTY;
        }
    }

    pos.castling=u.castling;
    pos.ep=u.ep;
    pos.halfmove=u.halfmove;

    if(pos.side==BLACK)
        pos.fullmove--;
}

static void generateLegal(vector<Move>& moves) {

    vector<Move> pseudo;

    generatePseudo(pseudo);

    moves.clear();

    for(const Move& m:pseudo) {

        Undo u;

        if(makeMove(m,u)) {

            moves.push_back(m);

            undoMove(m,u);
        }
    }
}

static int psq(int p,int sq) {

    int s=(p>=BP)?(sq^56):sq;

    int r=rankOf(s);
    int f=fileOf(s);

    switch(p%6) {

        case 1:
            return pstPawn[r*8+f];

        case 2: {
            static const int n[64]={
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,0,5,5,0,-20,-40,
                -30,5,10,15,15,10,5,-30,
                -30,0,15,20,20,15,0,-30,
                -30,5,15,20,20,15,5,-30,
                -30,0,10,15,15,10,0,-30,
                -40,-20,0,0,0,0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            };
            return n[s];
        }

        case 3:
            return 0;

        case 4:
            return 0;

        case 5:
            return 0;

        case 0:
            return 0;
    }

    return 0;
}

static int materialEvaluation() {

    int score=0;

    for(int sq=0;sq<64;sq++) {

        int p=pos.board[sq];

        if(!p)
            continue;

        int v=pieceValue(p);

        if(whitePiece(p))
            score+=v;
        else
            score-=v;

        int bonus=psq(p,sq);

        if(whitePiece(p))
            score+=bonus;
        else
            score-=bonus;
    }

    // Mobility
    int whiteMob=0;
    int blackMob=0;

    int oldSide=pos.side;

    pos.side=WHITE;

    vector<Move> wm;
    generatePseudo(wm);
    whiteMob=(int)wm.size();

    pos.side=BLACK;

    vector<Move> bm;
    generatePseudo(bm);
    blackMob=(int)bm.size();

    pos.side=oldSide;

    score+=(whiteMob-blackMob)*2;

    return pos.side==WHITE?score:-score;
}

// ============================================================
// NNUE
// ============================================================

static constexpr int NN_INPUT=768;
static constexpr int NN_H1=256;
static constexpr int NN_H2=32;

struct NNUE {

    bool loaded=false;

    vector<int16_t> w1;
    vector<int16_t> b1;

    vector<int16_t> w2;
    vector<int16_t> b2;

    vector<int16_t> wo;
    vector<int16_t> bo;

    bool load(const string& filename) {

        ifstream f(filename,ios::binary);

        if(!f)
            return false;

        char magic[7];

        f.read(magic,7);

        if(string(magic,7)!="SLNNUE2")
            return false;

        uint32_t in,h1,h2,out;

        f.read((char*)&in,4);
        f.read((char*)&h1,4);
        f.read((char*)&h2,4);
        f.read((char*)&out,4);

        if(in!=NN_INPUT ||
           h1!=NN_H1 ||
           h2!=NN_H2 ||
           out!=1)
            return false;

        w1.resize(NN_INPUT*NN_H1);
        b1.resize(NN_H1);

        w2.resize(NN_H1*NN_H2);
        b2.resize(NN_H2);

        wo.resize(NN_H2);
        bo.resize(1);

        f.read(
            (char*)w1.data(),
            w1.size()*sizeof(int16_t)
        );

        f.read(
            (char*)b1.data(),
            b1.size()*sizeof(int16_t)
        );

        f.read(
            (char*)w2.data(),
            w2.size()*sizeof(int16_t)
        );

        f.read(
            (char*)b2.data(),
            b2.size()*sizeof(int16_t)
        );

        f.read(
            (char*)wo.data(),
            wo.size()*sizeof(int16_t)
        );

        f.read(
            (char*)bo.data(),
            bo.size()*sizeof(int16_t)
        );

        loaded=f.good();

        return loaded;
    }

    int evaluate() const {

        if(!loaded)
            return materialEvaluation();

        alignas(32) float input[NN_INPUT]{};
        alignas(32) float h1[NN_H1]{};
        alignas(32) float h2[NN_H2]{};

        for(int sq=0;sq<64;sq++) {

            int p=pos.board[sq];

            if(p)
                input[(p-1)*64+sq]=1.0f;
        }

        for(int j=0;j<NN_H1;j++) {

            float sum=(float)b1[j]/256.0f;

            const int base=j*NN_INPUT;

            for(int i=0;i<NN_INPUT;i++)
                sum+=input[i]*(float)w1[base+i]/256.0f;

            h1[j]=max(0.0f,min(1.0f,sum));
        }

        for(int j=0;j<NN_H2;j++) {

            float sum=(float)b2[j]/256.0f;

            const int base=j*NN_H1;

            for(int i=0;i<NN_H1;i++)
                sum+=h1[i]*(float)w2[base+i]/256.0f;

            h2[j]=max(0.0f,min(1.0f,sum));
        }

        float result=(float)bo[0]/256.0f;

        for(int i=0;i<NN_H2;i++)
            result+=h2[i]*(float)wo[i]/256.0f;

        int score=(int)(result*1000.0f);

        return pos.side==WHITE?score:-score;
    }
};

static NNUE nnue;

// ============================================================
// Search
// ============================================================

static long long nowMs() {

    return chrono::duration_cast<
        chrono::milliseconds
    >(
        chrono::steady_clock::now().time_since_epoch()
    ).count();
}

static bool timeUp() {

    if(stopSearch.load())
        return true;

    if(stopTime && nowMs()>=stopTime) {
        stopSearch.store(true);
        return true;
    }

    return false;
}

static void scoreMoves(vector<Move>& moves,int ply) {

    for(auto &m:moves) {

        if(m.score)
            continue;

        if(killers[ply][0].from==m.from &&
           killers[ply][0].to==m.to) {

            m.score=90000;
            continue;
        }

        if(killers[ply][1].from==m.from &&
           killers[ply][1].to==m.to) {

            m.score=80000;
            continue;
        }

        m.score=
            historyTable[pos.side][m.from][m.to];
    }

    stable_sort(
        moves.begin(),
        moves.end(),
        [](const Move&a,const Move&b){
            return a.score>b.score;
        }
    );
}

static int quiescence(int alpha,int beta,int ply) {

    if(timeUp())
        return 0;

    int stand=nnue.evaluate();

    if(stand>=beta)
        return beta;

    if(stand>alpha)
        alpha=stand;

    vector<Move> moves;

    generateLegal(moves);

    for(auto &m:moves) {

        if(pos.board[m.to]==EMPTY &&
           !(m.flags&1))
            continue;

        Undo u;

        if(!makeMove(m,u))
            continue;

        int score=-quiescence(
            -beta,
            -alpha,
            ply+1
        );

        undoMove(m,u);

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
    int ply,
    bool allowNull
) {

    if(timeUp())
        return 0;

    if(ply>=MAX_PLY-1)
        return nnue.evaluate();

    bool check=inCheck(pos.side);

    if(depth<=0)
        return quiescence(alpha,beta,ply);

    uint64_t key=hashPosition();

    TTEntry &entry=TT[key&(TT_SIZE-1)];

    if(entry.key==key &&
       entry.depth>=depth) {

        if(entry.flag==0)
            return entry.score;

        if(entry.flag==1 &&
           entry.score<=alpha)
            return alpha;

        if(entry.flag==2 &&
           entry.score>=beta)
            return beta;
    }

    // Null move
    if(allowNull &&
       depth>=3 &&
       !check) {

        bool hasNonPawn=false;

        for(int s=0;s<64;s++) {

            int p=pos.board[s];

            if(ownPiece(p,pos.side) &&
               p!=WP &&
               p!=BP &&
               p!=WK &&
               p!=BK) {

                hasNonPawn=true;
                break;
            }
        }

        if(hasNonPawn) {

            int oldEP=pos.ep;
            pos.ep=-1;
            pos.side^=1;

            int score=-search(
                depth-3,
                -beta,
                -beta+1,
                ply+1,
                false
            );

            pos.side^=1;
            pos.ep=oldEP;

            if(score>=beta)
                return beta;
        }
    }

    vector<Move> moves;

    generateLegal(moves);

    if(moves.empty()) {

        if(check)
            return -MATE+ply;

        return 0;
    }

    scoreMoves(moves,ply);

    int bestScore=-INF;
    Move bestMove{};
    int legal=0;

    int originalAlpha=alpha;

    for(size_t i=0;i<moves.size();i++) {

        Move m=moves[i];

        Undo u;

        if(!makeMove(m,u))
            continue;

        legal++;

        int score;

        if(i==0) {

            score=-search(
                depth-1,
                -beta,
                -alpha,
                ply+1,
                true
            );

        } else {

            int reduction=0;

            if(depth>=3 &&
               i>=4 &&
               !check &&
               m.score<100000)
                reduction=1;

            score=-search(
                depth-1-reduction,
                -alpha-1,
                -alpha,
                ply+1,
                true
            );

            if(score>alpha &&
               score<beta) {

                score=-search(
                    depth-1,
                    -beta,
                    -alpha,
                    ply+1,
                    true
                );
            }
        }

        undoMove(m,u);

        if(score>bestScore) {

            bestScore=score;
            bestMove=m;
        }

        if(score>alpha)
            alpha=score;

        if(alpha>=beta) {

            if(!(m.score>=100000)) {

                killers[ply][1]=killers[ply][0];
                killers[ply][0]=m;

                historyTable[
                    pos.side
                ][m.from][m.to]+=depth*depth;
            }

            break;
        }
    }

    entry.key=key;
    entry.depth=depth;
    entry.score=bestScore;
    entry.best=bestMove;

    if(bestScore<=originalAlpha)
        entry.flag=1;
    else if(bestScore>=beta)
        entry.flag=2;
    else
        entry.flag=0;

    return bestScore;
}

static string moveToUCI(const Move&m) {

    string s;

    s.push_back(
        char('a'+fileOf(m.from))
    );

    s.push_back(
        char('1'+rankOf(m.from))
    );

    s.push_back(
        char('a'+fileOf(m.to))
    );

    s.push_back(
        char('1'+rankOf(m.to))
    );

    if(m.promotion) {

        char c='q';

        if(m.promotion==WR) c='r';
        if(m.promotion==WB) c='b';
        if(m.promotion==WN) c='n';

        s.push_back(c);
    }

    return s;
}

static int parseSquare(const string&s) {

    if(s.size()<2)
        return -1;

    int f=s[0]-'a';
    int r=s[1]-'1';

    if(f<0||f>7||r<0||r>7)
        return -1;

    return r*8+f;
}

static Move parseMove(const string&s) {

    Move m;

    if(s.size()<4)
        return m;

    m.from=parseSquare(s.substr(0,2));
    m.to=parseSquare(s.substr(2,2));

    if(s.size()>=5) {

        switch(s[4]) {

            case 'q': m.promotion=WQ; break;
            case 'r': m.promotion=WR; break;
            case 'b': m.promotion=WB; break;
            case 'n': m.promotion=WN; break;
        }
    }

    return m;
}

static void setStartPosition() {

    pos.board.fill(EMPTY);

    const string back="RNBQKBNR";

    for(int f=0;f<8;f++) {

        pos.board[f]=
            string("RNBQKBNR")[f]-'A'+1;

        pos.board[8+f]=WP;

        pos.board[48+f]=BP;

        pos.board[56+f]=
            string("RNBQKBNR")[f]-'A'+7;
    }

    // Correct piece encoding
    int whiteBack[8]={
        WR,WN,WB,WQ,WK,WB,WN,WR
    };

    int blackBack[8]={
        BR,BN,BB,BQ,BK,BB,BN,BR
    };

    for(int f=0;f<8;f++) {
        pos.board[f]=whiteBack[f];
        pos.board[56+f]=blackBack[f];
    }

    pos.side=WHITE;
    pos.castling=15;
    pos.ep=-1;
    pos.halfmove=0;
    pos.fullmove=1;
}

static void setFEN(const string&fen) {

    pos.board.fill(EMPTY);

    stringstream ss(fen);

    string board,side,castle,ep;
    ss>>board>>side>>castle>>ep;
    ss>>pos.halfmove>>pos.fullmove;

    int sq=56;

    for(char c:board) {

        if(c=='/') {
            sq-=16;
            continue;
        }

        if(c>='1'&&c<='8') {
            sq+=c-'0';
            continue;
        }

        int p=EMPTY;

        switch(c) {

            case 'P':p=WP;break;
            case 'N':p=WN;break;
            case 'B':p=WB;break;
            case 'R':p=WR;break;
            case 'Q':p=WQ;break;
            case 'K':p=WK;break;

            case 'p':p=BP;break;
            case 'n':p=BN;break;
            case 'b':p=BB;break;
            case 'r':p=BR;break;
            case 'q':p=BQ;break;
            case 'k':p=BK;break;
        }

        pos.board[sq++]=p;
    }

    pos.side=(side=="w"?WHITE:BLACK);

    pos.castling=0;

    if(castle.find('K')!=string::npos)
        pos.castling|=1;

    if(castle.find('Q')!=string::npos)
        pos.castling|=2;

    if(castle.find('k')!=string::npos)
        pos.castling|=4;

    if(castle.find('q')!=string::npos)
        pos.castling|=8;

    pos.ep=-1;

    if(ep!="-")
        pos.ep=parseSquare(ep);
}

static void applyUCIMoves(const vector<string>&moves) {

    for(const string&text:moves) {

        Move wanted=parseMove(text);

        vector<Move> legal;
        generateLegal(legal);

        for(const Move&m:legal) {

            if(m.from==wanted.from &&
               m.to==wanted.to &&
               m.promotion==wanted.promotion) {

                Undo u;

                if(makeMove(m,u))
                    break;
            }
        }
    }
}

static Move findBestMove(int depth) {

    Move best{};

    for(int d=1;d<=depth;d++) {

        if(timeUp())
            break;

        int score=search(
            d,
            -INF,
            INF,
            0,
            true
        );

        uint64_t key=hashPosition();

        TTEntry&e=TT[key&(TT_SIZE-1)];

        if(e.key==key) {

            best=e.best;

            cout
                <<"info depth "<<d
                <<" score cp "<<score
                <<" nodes 0"
                <<" pv "<<moveToUCI(best)
                <<"\n";
        }

        cout.flush();
    }

    return best;
}

static void handlePosition(const string&line) {

    stringstream ss(line);

    string token;
    ss>>token;

    ss>>token;

    if(token=="startpos") {

        setStartPosition();

        ss>>token;

    } else if(token=="fen") {

        string fen;

        for(int i=0;i<6;i++) {

            string x;
            ss>>x;

            if(i)
                fen+=" ";

            fen+=x;
        }

        setFEN(fen);

        ss>>token;
    }

    if(token=="moves") {

        vector<string> moves;

        while(ss>>token)
            moves.push_back(token);

        applyUCIMoves(moves);
    }
}

static void handleGo(const string&line) {

    stopSearch.store(false);

    stopTime=0;

    stringstream ss(line);

    string t;

    int depth=0;
    int movetime=0;

    while(ss>>t) {

        if(t=="depth") {

            ss>>depth;

        } else if(t=="movetime") {

            ss>>movetime;
        }
    }

    if(depth<=0)
        depth=maxDepth;

    if(movetime>0)
        stopTime=nowMs()+movetime;

    Move best=findBestMove(depth);

    cout<<"bestmove "<<moveToUCI(best)<<"\n";
    cout.flush();
}

int main() {

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    initZobrist();

    setStartPosition();

    // NNUE file beside executable
    if(nnue.load("simplelogics.nnue")) {

        cerr<<"NNUE loaded\n";

    } else {

        cerr<<"WARNING: simplelogics.nnue not loaded\n";
        cerr<<"Using fallback evaluation\n";
    }

    string line;

    while(getline(cin,line)) {

        if(line=="uci") {

            cout<<"id name SimpleLogics NNUE\n";
            cout<<"id author SimpleLogics\n";
            cout<<"option name EvalFile type string default simplelogics.nnue\n";
            cout<<"uciok\n";
            cout.flush();

        } else if(line=="isready") {

            cout<<"readyok\n";
            cout.flush();

        } else if(line.rfind("setoption",0)==0) {

            // NNUE is loaded from the executable directory.

        } else if(line.rfind("position",0)==0) {

            handlePosition(line);

        } else if(line.rfind("go",0)==0) {

            handleGo(line);

        } else if(line=="stop") {

            stopSearch.store(true);

        } else if(line=="ucinewgame") {

            TT.assign(TT.size(),TTEntry{});

        } else if(line=="quit") {

            break;
        }
    }

    return 0;
}
