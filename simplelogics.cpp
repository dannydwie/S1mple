#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

enum Piece {
    EMPTY=0,
    WP=1, WN=2, WB=3, WR=4, WQ=5, WK=6,
    BP=7, BN=8, BB=9, BR=10, BQ=11, BK=12
};

struct Move {
    int from, to, promo;
};

struct Position {
    array<int,64> b{};
    bool white=true;
};

static int value(int p) {
    static const int v[]={
        0,100,320,330,500,900,20000,
        100,320,330,500,900,20000
    };
    return v[p];
}

static bool whitePiece(int p) { return p>=WP && p<=WK; }
static bool blackPiece(int p) { return p>=BP && p<=BK; }

static bool own(int p,bool w) {
    return w ? whitePiece(p) : blackPiece(p);
}

static bool enemy(int p,bool w) {
    return w ? blackPiece(p) : whitePiece(p);
}

static int fileOf(int s){ return s&7; }
static int rankOf(int s){ return s>>3; }

static void addMove(vector<Move>& m,int f,int t,int p=0) {
    m.push_back({f,t,p});
}

static void generate(Position& p, vector<Move>& moves) {
    moves.clear();

    const bool w=p.white;

    for(int s=0;s<64;s++) {
        int pc=p.b[s];
        if(!own(pc,w)) continue;

        int r=rankOf(s), f=fileOf(s);

        if(pc==WP || pc==BP) {
            int d=w?8:-8;
            int nr=r+(w?1:-1);

            if(nr>=0 && nr<8) {
                int t=s+d;

                if(p.b[t]==EMPTY) {
                    if(nr==0 || nr==7) {
                        addMove(moves,s,t,w?WQ:BQ);
                        addMove(moves,s,t,w?WR:BR);
                        addMove(moves,s,t,w?WB:BB);
                        addMove(moves,s,t,w?WN:BN);
                    } else addMove(moves,s,t);

                    int start=w?1:6;
                    if(r==start && p.b[s+2*d]==EMPTY)
                        addMove(moves,s,s+2*d);
                }

                for(int df:{-1,1}) {
                    int nf=f+df;
                    if(nf<0||nf>7) continue;
                    int t2=t+df;

                    if(t2>=0&&t2<64&&enemy(p.b[t2],w)) {
                        if(nr==0||nr==7) {
                            addMove(moves,s,t2,w?WQ:BQ);
                            addMove(moves,s,t2,w?WR:BR);
                            addMove(moves,s,t2,w?WB:BB);
                            addMove(moves,s,t2,w?WN:BN);
                        } else addMove(moves,s,t2);
                    }
                }
            }
        }

        if(pc==WN || pc==BN) {
            static const int dr[]={2,2,1,1,-1,-1,-2,-2};
            static const int df[]={1,-1,2,-2,2,-2,1,-1};

            for(int i=0;i<8;i++) {
                int nr=r+dr[i], nf=f+df[i];
                if(nr<0||nr>7||nf<0||nf>7) continue;
                int t=nr*8+nf;
                if(!own(p.b[t],w)) addMove(moves,s,t);
            }
        }

        if(pc==WK || pc==BK) {
            for(int dr=-1;dr<=1;dr++)
                for(int df=-1;df<=1;df++) {
                    if(!dr&&!df) continue;
                    int nr=r+dr,nf=f+df;
                    if(nr<0||nr>7||nf<0||nf>7) continue;
                    int t=nr*8+nf;
                    if(!own(p.b[t],w)) addMove(moves,s,t);
                }
        }

        bool bishop=(pc==WB||pc==BB||pc==WQ||pc==BQ);
        bool rook=(pc==WR||pc==BR||pc==WQ||pc==BQ);

        if(bishop||rook) {
            static const int dirs[8][2]={
                {1,0},{-1,0},{0,1},{0,-1},
                {1,1},{1,-1},{-1,1},{-1,-1}
            };

            for(int i=0;i<8;i++) {
                bool diag=dirs[i][0]!=0 && dirs[i][1]!=0;
                if(diag&&!bishop) continue;
                if(!diag&&!rook) continue;

                int nr=r+dirs[i][0];
                int nf=f+dirs[i][1];

                while(nr>=0&&nr<8&&nf>=0&&nf<8) {
                    int t=nr*8+nf;

                    if(p.b[t]==EMPTY)
                        addMove(moves,s,t);
                    else {
                        if(enemy(p.b[t],w))
                            addMove(moves,s,t);
                        break;
                    }

                    nr+=dirs[i][0];
                    nf+=dirs[i][1];
                }
            }
        }
    }
}

static int evaluate(const Position& p) {
    int score=0;

    static const int pst[6][8]={
        {0,5,5,10,10,5,5,0},
        {-5,0,5,10,10,5,0,-5},
        {-5,5,5,10,10,5,5,-5},
        {0,0,5,10,10,5,0,0},
        {0,0,0,5,5,0,0,0},
        {0,0,5,10,10,5,0,0}
    };

    for(int s=0;s<64;s++) {
        int pc=p.b[s];
        if(!pc) continue;

        int v=value(pc);

        if(pc<=WK) {
            score+=v;
            if(pc==WP) score+=pst[0][rankOf(s)];
            if(pc==WN) score+=pst[1][rankOf(s)];
            if(pc==WB) score+=pst[2][rankOf(s)];
            if(pc==WR) score+=pst[3][rankOf(s)];
            if(pc==WQ) score+=pst[4][rankOf(s)];
        } else {
            score-=v;
            int rr=7-rankOf(s);
            if(pc==BP) score-=pst[0][rr];
            if(pc==BN) score-=pst[1][rr];
            if(pc==BB) score-=pst[2][rr];
            if(pc==BR) score-=pst[3][rr];
            if(pc==BQ) score-=pst[4][rr];
        }
    }

    return p.white ? score : -score;
}

static Position startPosition() {
    Position p;
    p.b.fill(EMPTY);

    const int backW[]={WR,WN,WB,WQ,WK,WB,WN,WR};
    const int backB[]={BR,BN,BB,BQ,BK,BB,BN,BR};

    for(int i=0;i<8;i++) {
        p.b[i]=backW[i];
        p.b[8+i]=WP;
        p.b[48+i]=BP;
        p.b[56+i]=backB[i];
    }

    p.white=true;
    return p;
}

static Position makeMove(const Position& p,const Move& m) {
    Position n=p;
    int pc=n.b[m.from];

    n.b[m.from]=EMPTY;
    n.b[m.to]=m.promo?m.promo:pc;
    n.white=!p.white;

    return n;
}

static int negamax(const Position& p,int depth,int alpha,int beta) {
    if(depth<=0)
        return evaluate(p);

    vector<Move> moves;
    generate(const_cast<Position&>(p),moves);

    if(moves.empty())
        return evaluate(p);

    sort(moves.begin(),moves.end(),
        [&](const Move&a,const Move&b) {
            return value(p.b[a.to])>value(p.b[b.to]);
        });

    int best=-numeric_limits<int>::max();

    for(const auto&m:moves) {
        Position n=makeMove(p,m);
        int sc=-negamax(n,depth-1,-beta,-alpha);

        best=max(best,sc);
        alpha=max(alpha,sc);

        if(alpha>=beta)
            break;
    }

    return best;
}

static Move searchBest(Position p,int depth) {
    vector<Move> moves;
    generate(p,moves);

    Move best=moves.empty()?Move{0,0,0}:moves[0];
    int bestScore=-numeric_limits<int>::max();

    for(auto&m:moves) {
        Position n=makeMove(p,m);
        int score=-negamax(n,depth-1,
                           -numeric_limits<int>::max(),
                           numeric_limits<int>::max());

        if(score>bestScore) {
            bestScore=score;
            best=m;
        }
    }

    return best;
}

static string square(int s) {
    string r;
    r+=char('a'+fileOf(s));
    r+=char('1'+rankOf(s));
    return r;
}

static string uciMove(const Move&m) {
    string s=square(m.from)+square(m.to);

    if(m.promo) {
        int p=m.promo;
        char c='q';

        if(p==WN||p==BN)c='n';
        if(p==WB||p==BB)c='b';
        if(p==WR||p==BR)c='r';

        s+=c;
    }

    return s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Position pos=startPosition();

    string line;

    while(getline(cin,line)) {
        if(line=="uci") {
            cout<<"id name SimpleLogics"<<'\n';
            cout<<"id author Danny"<<'\n';
            cout<<"option name Skill Level type spin default 20 min 1 max 20"<<'\n';
            cout<<"uciok"<<'\n';
        }

        else if(line=="isready") {
            cout<<"readyok"<<'\n';
        }

        else if(line=="ucinewgame") {
            pos=startPosition();
        }

        else if(line.rfind("position startpos",0)==0) {
            pos=startPosition();

            size_t x=line.find("moves");

            if(x!=string::npos) {
                stringstream ss(line.substr(x+5));
                string ms;

                while(ss>>ms) {
                    if(ms.size()<4) continue;

                    int f=(ms[0]-'a')+(ms[1]-'1')*8;
                    int t=(ms[2]-'a')+(ms[3]-'1')*8;

                    vector<Move> moves;
                    generate(pos,moves);

                    for(auto&m:moves) {
                        if(m.from==f&&m.to==t) {
                            pos=makeMove(pos,m);
                            break;
                        }
                    }
                }
            }
        }

        else if(line.rfind("go",0)==0) {
            Move best=searchBest(pos,5);

            cout<<"bestmove "<<uciMove(best)<<'\n';
        }

        else if(line=="quit") {
            break;
        }

        cout.flush();
    }

    return 0;
}
