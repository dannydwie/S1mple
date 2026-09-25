#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nnue_data.h"

using namespace std;

enum Piece {
    EMPTY=0,
    WP,WN,WB,WR,WQ,WK,
    BP,BN,BB,BR,BQ,BK
};

enum { WHITE=0, BLACK=1 };

static constexpr int INF=32000;
static constexpr int MATE=30000;
static constexpr int MAX_PLY=128;

struct Move {
    int from=0,to=0,promotion=0,flags=0,score=0;
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
static atomic<bool> stopSearch(false);

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

static inline bool whitePiece(int p) {
    return p>=WP && p<=WK;
}

static inline bool blackPiece(int p) {
    return p>=BP && p<=BK;
}

static inline bool ownPiece(int p,int s) {
    return s==WHITE ? whitePiece(p) : blackPiece(p);
}

static inline bool enemyPiece(int p,int s) {
    return s==WHITE ? blackPiece(p) : whitePiece(p);
}

static inline int fileOf(int s) {
    return s&7;
}

static inline int rankOf(int s) {
    return s>>3;
}

static int pieceValue(int p) {

    switch(p) {
        case WP: case BP: return 100;
        case WN: case BN: return 320;
        case WB: case BB: return 330;
        case WR: case BR: return 500;
        case WQ: case BQ: return 900;
        case WK: case BK: return 20000;
    }

    return 0;
}

static bool attacked(int sq,int side) {

    int r=rankOf(sq);
    int f=fileOf(sq);

    if(side==WHITE) {

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

    static const int knr[8]={
        -2,-2,-1,-1,1,1,2,2
    };

    static const int knf[8]={
        -1,1,-2,2,-2,2,-1,1
    };

    for(int i=0;i<8;i++) {

        int rr=r+knr[i];
        int ff=f+knf[i];

        if(rr>=0&&rr<8&&ff>=0&&ff<8) {

            if(pos.board[rr*8+ff]==
               (side==WHITE?WN:BN))
                return true;
        }
    }

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

                if(p==(side==WHITE?WB:BB) ||
                   p==(side==WHITE?WQ:BQ))
                    return true;

                break;
            }

            rr+=drB[d];
            ff+=dfB[d];
        }
    }

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

                if(p==(side==WHITE?WR:BR) ||
                   p==(side==WHITE?WQ:BQ))
                    return true;

                break;
            }

            rr+=drR[d];
            ff+=dfR[d];
        }
    }

    for(int rr=max(0,r-1);rr<=min(7,r+1);rr++)
        for(int ff=max(0,f-1);ff<=min(7,f+1);ff++)
            if((rr!=r||ff!=f) &&
               pos.board[rr*8+ff]==
               (side==WHITE?WK:BK))
                return true;

    return false;
}

static bool inCheck(int side) {

    int king=side==WHITE?WK:BK;

    for(int s=0;s<64;s++)
        if(pos.board[s]==king)
            return attacked(s,side^1);

    return true;
}

static void addMove(
    vector<Move>&m,
    int from,
    int to,
    int promo=0,
    int flags=0
) {

    Move x;

    x.from=from;
    x.to=to;
    x.promotion=promo;
    x.flags=flags;

    int cap=pos.board[to];

    if(flags&1)
        cap=pos.side==WHITE?BP:WP;

    if(cap)
        x.score=100000+
            pieceValue(cap)*10-
            pieceValue(pos.board[from]);

    m.push_back(x);
}

static void generatePseudo(vector<Move>&m) {

    m.clear();

    int s=pos.side;

    for(int sq=0;sq<64;sq++) {

        int p=pos.board[sq];

        if(!ownPiece(p,s))
            continue;

        int r=rankOf(sq);
        int f=fileOf(sq);

        if(p==WP||p==BP) {

            int dir=p==WP?1:-1;
            int start=p==WP?1:6;
            int pr=p==WP?7:0;

            int nr=r+dir;

            if(nr>=0&&nr<8) {

                int to=nr*8+f;

                if(pos.board[to]==EMPTY) {

                    if(nr==pr) {

                        addMove(m,sq,to,WQ);
                        addMove(m,sq,to,WR);
                        addMove(m,sq,to,WB);
                        addMove(m,sq,to,WN);

                    } else {

                        addMove(m,sq,to);

                        if(r==start) {

                            int to2=(r+2*dir)*8+f;

                            if(pos.board[to2]==EMPTY)
                                addMove(m,sq,to2,0,2);
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

                if(enemyPiece(pos.board[to],s)) {

                    if(rr==pr) {

                        addMove(m,sq,to,WQ);
                        addMove(m,sq,to,WR);
                        addMove(m,sq,to,WB);
                        addMove(m,sq,to,WN);

                    } else
                        addMove(m,sq,to);
                }

                if(to==pos.ep)
                    addMove(m,sq,to,0,1);
            }

            continue;
        }

        if(p==WN||p==BN) {

            static const int dr[8]={
                -2,-2,-1,-1,1,1,2,2
            };

            static const int df[8]={
                -1,1,-2,2,-2,2,-1,1
            };

            for(int i=0;i<8;i++) {

                int rr=r+dr[i];
                int ff=f+df[i];

                if(rr>=0&&rr<8&&ff>=0&&ff<8) {

                    int to=rr*8+ff;

                    if(!ownPiece(pos.board[to],s))
                        addMove(m,sq,to);
                }
            }

            continue;
        }

        if(p==WB||p==BB||
           p==WR||p==BR||
           p==WQ||p==BQ) {

            static const int dr[8]={
                1,-1,0,0,1,1,-1,-1
            };

            static const int df[8]={
                0,0,1,-1,1,-1,1,-1
            };

            int begin=0,end=8;

            if(p==WB||p==BB)
                begin=4;

            if(p==WR||p==BR)
                end=4;

            for(int d=begin;d<end;d++) {

                int rr=r+dr[d];
                int ff=f+df[d];

                while(rr>=0&&rr<8&&ff>=0&&ff<8) {

                    int to=rr*8+ff;

                    if(ownPiece(pos.board[to],s))
                        break;

                    addMove(m,sq,to);

                    if(pos.board[to])
                        break;

                    rr+=dr[d];
                    ff+=df[d];
                }
            }

            continue;
        }

        if(p==WK||p==BK) {

            for(int rr=max(0,r-1);rr<=min(7,r+1);rr++)
                for(int ff=max(0,f-1);ff<=min(7,f+1);ff++) {

                    if(rr==r&&ff==f)
                        continue;

                    int to=rr*8+ff;

                    if(!ownPiece(pos.board[to],s))
                        addMove(m,sq,to);
                }
        }
    }
}

static bool makeMove(const Move&m,Undo&u) {

    u.captured=pos.board[m.to];
    u.castling=pos.castling;
    u.ep=pos.ep;
    u.halfmove=pos.halfmove;

    if(m.flags&1) {

        int csq=pos.side==WHITE?m.to-8:m.to+8;

        u.captured=pos.board[csq];
        pos.board[csq]=EMPTY;
    }

    int p=pos.board[m.from];

    pos.board[m.from]=EMPTY;

    if(m.promotion)
        pos.board[m.to]=
            pos.side==WHITE?m.promotion:m.promotion+6;
    else
        pos.board[m.to]=p;

    if(p==WK) pos.castling&=~3;
    if(p==BK) pos.castling&=~12;

    pos.ep=-1;

    if(m.flags&2)
        pos.ep=pos.side==WHITE?
            m.from+8:m.from-8;

    if(p==WP||p==BP||u.captured)
        pos.halfmove=0;
    else
        pos.halfmove++;

    pos.side^=1;

    if(inCheck(pos.side^1)) {

        pos.side^=1;

        pos.board[m.from]=p;
        pos.board[m.to]=u.captured;

        if(m.flags&1) {

            int csq=pos.side==WHITE?
                m.to-8:m.to+8;

            pos.board[csq]=
                pos.side==WHITE?BP:WP;
        }

        pos.castling=u.castling;
        pos.ep=u.ep;
        pos.halfmove=u.halfmove;

        return false;
    }

    return true;
}

static void undoMove(const Move&m,const Undo&u) {

    pos.side^=1;

    int p=pos.board[m.to];

    if(m.promotion)
        p=pos.side==WHITE?WP:BP;

    pos.board[m.from]=p;
    pos.board[m.to]=u.captured;

    if(m.flags&1) {

        int csq=pos.side==WHITE?
            m.to-8:m.to+8;

        pos.board[csq]=
            pos.side==WHITE?BP:WP;
    }

    pos.castling=u.castling;
    pos.ep=u.ep;
    pos.halfmove=u.halfmove;
}

static void legalMoves(vector<Move>&out) {

    vector<Move> p;
    generatePseudo(p);

    out.clear();

    for(auto&m:p) {

        Undo u;

        if(makeMove(m,u)) {

            out.push_back(m);
            undoMove(m,u);
        }
    }
}

// ============================================================
// Embedded NNUE
// ============================================================

static int nnueEvaluate() {

    /*
       nnue_data.h must contain:

       SL_W1[]
       SL_B1[]
       SL_W2[]
       SL_B2[]
       SL_WO[]
       SL_BO[]

       generated automatically by build.yml.
    */

    float h1[256]{};
    float h2[32]{};

    float input[768]{};

    for(int sq=0;sq<64;sq++) {

        int p=pos.board[sq];

        if(p)
            input[(p-1)*64+sq]=1.0f;
    }

    for(int j=0;j<256;j++) {

        float x=
            (float)SL_B1[j]/256.0f;

        for(int i=0;i<768;i++)
            x+=input[i]*
               (float)SL_W1[j*768+i]/256.0f;

        h1[j]=max(0.0f,min(1.0f,x));
    }

    for(int j=0;j<32;j++) {

        float x=
            (float)SL_B2[j]/256.0f;

        for(int i=0;i<256;i++)
            x+=h1[i]*
               (float)SL_W2[j*256+i]/256.0f;

        h2[j]=max(0.0f,min(1.0f,x));
    }

    float result=
        (float)SL_BO[0]/256.0f;

    for(int i=0;i<32;i++)
        result+=h2[i]*
            (float)SL_WO[i]/256.0f;

    int score=(int)(result*1000.0f);

    return pos.side==WHITE?score:-score;
}

// ============================================================
// Search
// ============================================================

static int quiescence(int alpha,int beta,int ply) {

    if(stopSearch)
        return 0;

    int stand=nnueEvaluate();

    if(stand>=beta)
        return beta;

    if(stand>alpha)
        alpha=stand;

    vector<Move> moves;
    legalMoves(moves);

    for(auto&m:moves) {

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
    int ply
) {

    if(stopSearch)
        return 0;

    if(depth<=0)
        return quiescence(
            alpha,beta,ply
        );

    vector<Move> moves;

    legalMoves(moves);

    if(moves.empty()) {

        if(inCheck(pos.side))
            return -MATE+ply;

        return 0;
    }

    uint64_t key=0;

    for(int s=0;s<64;s++)
        key^=((uint64_t)
            (pos.board[s]+1)*
            0x9e3779b97f4a7c15ULL)
            <<(s&31);

    key^=pos.side;

    TTEntry &e=TT[key&(TT_SIZE-1)];

    if(e.key==key&&e.depth>=depth) {

        if(e.flag==0)
            return e.score;

        if(e.flag==1&&e.score<=alpha)
            return alpha;

        if(e.flag==2&&e.score>=beta)
            return beta;
    }

    sort(
        moves.begin(),
        moves.end(),
        [](const Move&a,const Move&b){
            return a.score>b.score;
        }
    );

    int originalAlpha=alpha;
    int best=-INF;
    Move bestMove{};

    for(size_t i=0;i<moves.size();i++) {

        Undo u;

        if(!makeMove(moves[i],u))
            continue;

        int score;

        if(i==0) {

            score=-search(
                depth-1,
                -beta,
                -alpha,
                ply+1
            );

        } else {

            score=-search(
                depth-1,
                -alpha-1,
                -alpha,
                ply+1
            );

            if(score>alpha&&score<beta)
                score=-search(
                    depth-1,
                    -beta,
                    -alpha,
                    ply+1
                );
        }

        undoMove(moves[i],u);

        if(score>best) {

            best=score;
            bestMove=moves[i];
        }

        if(score>alpha)
            alpha=score;

        if(alpha>=beta)
            break;
    }

    e.key=key;
    e.depth=depth;
    e.score=best;
    e.best=bestMove;

    if(best<=originalAlpha)
        e.flag=1;
    else if(best>=beta)
        e.flag=2;
    else
        e.flag=0;

    return best;
}

static string uciMove(const Move&m) {

    string s;

    s+=char('a'+fileOf(m.from));
    s+=char('1'+rankOf(m.from));
    s+=char('a'+fileOf(m.to));
    s+=char('1'+rankOf(m.to));

    if(m.promotion) {

        char c='q';

        if(m.promotion==WR)c='r';
        if(m.promotion==WB)c='b';
        if(m.promotion==WN)c='n';

        s+=c;
    }

    return s;
}

static void startpos() {

    pos.board.fill(EMPTY);

    int back[8]={
        WR,WN,WB,WQ,WK,WB,WN,WR
    };

    int backB[8]={
        BR,BN,BB,BQ,BK,BB,BN,BR
    };

    for(int i=0;i<8;i++) {

        pos.board[i]=back[i];
        pos.board[8+i]=WP;

        pos.board[48+i]=BP;
        pos.board[56+i]=backB[i];
    }

    pos.side=WHITE;
    pos.castling=15;
    pos.ep=-1;
    pos.halfmove=0;
    pos.fullmove=1;
}

static int sqFrom(const string&s) {

    if(s.size()<2)
        return -1;

    int f=s[0]-'a';
    int r=s[1]-'1';

    if(f<0||f>7||r<0||r>7)
        return -1;

    return r*8+f;
}

static void setFEN(const string&fen) {

    pos.board.fill(EMPTY);

    string a,b,c,d;
    stringstream ss(fen);

    ss>>a>>b>>c>>d;

    int sq=56;

    for(char x:a) {

        if(x=='/') {
            sq-=16;
            continue;
        }

        if(x>='1'&&x<='8') {
            sq+=x-'0';
            continue;
        }

        int p=EMPTY;

        switch(x) {

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

    pos.side=b=="w"?WHITE:BLACK;

    pos.castling=0;

    if(c.find('K')!=string::npos)
        pos.castling|=1;

    if(c.find('Q')!=string::npos)
        pos.castling|=2;

    if(c.find('k')!=string::npos)
        pos.castling|=4;

    if(c.find('q')!=string::npos)
        pos.castling|=8;

    pos.ep=-1;

    if(d!="-")
        pos.ep=sqFrom(d);
}

static void applyMoveString(const string&s) {

    if(s.size()<4)
        return;

    int from=sqFrom(s.substr(0,2));
    int to=sqFrom(s.substr(2,2));

    int promo=0;

    if(s.size()>=5) {

        if(s[4]=='q')promo=WQ;
        if(s[4]=='r')promo=WR;
        if(s[4]=='b')promo=WB;
        if(s[4]=='n')promo=WN;
    }

    vector<Move> moves;
    legalMoves(moves);

    for(auto&m:moves) {

        if(m.from==from&&
           m.to==to&&
           m.promotion==promo) {

            Undo u;

            makeMove(m,u);
            return;
        }
    }
}

static void positionCommand(const string&line) {

    stringstream ss(line);

    string x;

    ss>>x;
    ss>>x;

    if(x=="startpos") {

        startpos();

        ss>>x;

    } else if(x=="fen") {

        string fen;

        for(int i=0;i<6;i++) {

            string z;
            ss>>z;

            if(i)fen+=" ";

            fen+=z;
        }

        setFEN(fen);

        ss>>x;
    }

    if(x=="moves") {

        while(ss>>x)
            applyMoveString(x);
    }
}

static void goCommand(const string&line) {

    stopSearch=false;

    stringstream ss(line);

    string x;

    int depth=6;

    while(ss>>x) {

        if(x=="depth")
            ss>>depth;
    }

    vector<Move> moves;

    legalMoves(moves);

    if(moves.empty()) {

        cout<<"bestmove 0000\n";
        return;
    }

    Move best=moves[0];

    for(int d=1;d<=depth;d++) {

        int score=search(
            d,
            -INF,
            INF,
            0
        );

        uint64_t key=0;

        for(int s=0;s<64;s++)
            key^=((uint64_t)
                (pos.board[s]+1)*
                0x9e3779b97f4a7c15ULL)
                <<(s&31);

        key^=pos.side;

        TTEntry&e=TT[key&(TT_SIZE-1)];

        if(e.key==key)
            best=e.best;

        cout<<"info depth "
            <<d
            <<" score cp "
            <<score
            <<" pv "
            <<uciMove(best)
            <<"\n";

        cout.flush();
    }

    cout<<"bestmove "
        <<uciMove(best)
        <<"\n";

    cout.flush();
}

int main() {

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    startpos();

    string line;

    while(getline(cin,line)) {

        if(line=="uci") {

            cout<<"id name SimpleLogics NNUE\n";
            cout<<"id author SimpleLogics\n";
            cout<<"uciok\n";

        } else if(line=="isready") {

            cout<<"readyok\n";

        } else if(line=="ucinewgame") {

            TT.assign(TT.size(),TTEntry{});

        } else if(line.rfind("position",0)==0) {

            positionCommand(line);

        } else if(line.rfind("go",0)==0) {

            goCommand(line);

        } else if(line=="stop") {

            stopSearch=true;

        } else if(line=="quit") {

            break;
        }

        cout.flush();
    }

    return 0;
}
