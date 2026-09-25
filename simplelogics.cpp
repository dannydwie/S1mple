#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

enum Piece {
    EMPTY=0,
    WP,WN,WB,WR,WQ,WK,
    BP,BN,BB,BR,BQ,BK
};

struct Move {
    int from;
    int to;
    int promo;
    bool castle;
    bool ep;
};

struct Position {
    array<int,64> b{};
    bool white=true;

    bool cwk=true;
    bool cwq=true;
    bool cbk=true;
    bool cbq=true;

    int ep=-1;
};

static const int INF  = 100000000;
static const int MATE = 1000000;

static int pieceValue(int p) {
    static const int v[]={
        0,
        100,320,330,500,900,20000,
        100,320,330,500,900,20000
    };
    return v[p];
}

static bool isWhite(int p) {
    return p>=WP && p<=WK;
}

static bool isBlack(int p) {
    return p>=BP && p<=BK;
}

static bool ownPiece(int p,bool white) {
    return white ? isWhite(p) : isBlack(p);
}

static bool enemyPiece(int p,bool white) {
    return white ? isBlack(p) : isWhite(p);
}

static int fileOf(int s) {
    return s&7;
}

static int rankOf(int s) {
    return s>>3;
}

static int squareOf(char file,char rank) {
    return (file-'a')+(rank-'1')*8;
}

static Position startPosition() {
    Position p;
    p.b.fill(EMPTY);

    int whiteBack[8]={
        WR,WN,WB,WQ,WK,WB,WN,WR
    };

    int blackBack[8]={
        BR,BN,BB,BQ,BK,BB,BN,BR
    };

    for(int i=0;i<8;i++) {
        p.b[i]=whiteBack[i];
        p.b[8+i]=WP;

        p.b[48+i]=BP;
        p.b[56+i]=blackBack[i];
    }

    p.white=true;
    p.cwk=p.cwq=p.cbk=p.cbq=true;
    p.ep=-1;

    return p;
}

static int kingSquare(const Position& p,bool white) {
    int king=white?WK:BK;

    for(int s=0;s<64;s++)
        if(p.b[s]==king)
            return s;

    return -1;
}

static bool squareAttacked(
    const Position& p,
    int sq,
    bool byWhite
) {
    int r=rankOf(sq);
    int f=fileOf(sq);

    /* pawns */

    int pawn=byWhite?WP:BP;
    int pr=byWhite?r-1:r+1;

    if(pr>=0 && pr<8) {
        for(int df : {-1,1}) {
            int nf=f+df;

            if(nf>=0 && nf<8) {
                if(p.b[pr*8+nf]==pawn)
                    return true;
            }
        }
    }

    /* knights */

    int knight=byWhite?WN:BN;

    const int knr[8]={
        2,2,1,1,-1,-1,-2,-2
    };

    const int knf[8]={
        1,-1,2,-2,2,-2,1,-1
    };

    for(int i=0;i<8;i++) {
        int nr=r+knr[i];
        int nf=f+knf[i];

        if(nr>=0 && nr<8 &&
           nf>=0 && nf<8) {

            if(p.b[nr*8+nf]==knight)
                return true;
        }
    }

    /* king */

    int king=byWhite?WK:BK;

    for(int dr=-1;dr<=1;dr++) {
        for(int df=-1;df<=1;df++) {

            if(dr==0 && df==0)
                continue;

            int nr=r+dr;
            int nf=f+df;

            if(nr>=0 && nr<8 &&
               nf>=0 && nf<8) {

                if(p.b[nr*8+nf]==king)
                    return true;
            }
        }
    }

    /* bishops / queens */

    int bishop=byWhite?WB:BB;
    int rook=byWhite?WR:BR;
    int queen=byWhite?WQ:BQ;

    const int dirs[8][2]={
        {1,1},
        {1,-1},
        {-1,1},
        {-1,-1},
        {1,0},
        {-1,0},
        {0,1},
        {0,-1}
    };

    for(int i=0;i<8;i++) {

        int nr=r+dirs[i][0];
        int nf=f+dirs[i][1];

        while(nr>=0 && nr<8 &&
              nf>=0 && nf<8) {

            int pc=p.b[nr*8+nf];

            if(pc!=EMPTY) {

                if(i<4) {
                    if(pc==bishop || pc==queen)
                        return true;
                } else {
                    if(pc==rook || pc==queen)
                        return true;
                }

                break;
            }

            nr+=dirs[i][0];
            nf+=dirs[i][1];
        }
    }

    return false;
}

static bool inCheck(
    const Position& p,
    bool white
) {
    int k=kingSquare(p,white);

    if(k<0)
        return true;

    return squareAttacked(p,k,!white);
}

static void addMove(
    vector<Move>& moves,
    int from,
    int to,
    int promo=0,
    bool castle=false,
    bool ep=false
) {
    moves.push_back({
        from,to,promo,castle,ep
    });
}

static void generatePseudo(
    const Position& p,
    vector<Move>& moves
) {
    moves.clear();

    bool w=p.white;

    for(int s=0;s<64;s++) {

        int pc=p.b[s];

        if(!ownPiece(pc,w))
            continue;

        int r=rankOf(s);
        int f=fileOf(s);

        /* PAWN */

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

                    } else {
                        addMove(moves,s,t);
                    }

                    int start=w?1:6;

                    if(r==start &&
                       p.b[s+2*d]==EMPTY) {

                        addMove(
                            moves,
                            s,
                            s+2*d
                        );
                    }
                }

                for(int df : {-1,1}) {

                    int nf=f+df;

                    if(nf<0 || nf>7)
                        continue;

                    int t2=t+df;

                    if(t2<0 || t2>=64)
                        continue;

                    bool capture=
                        enemyPiece(p.b[t2],w);

                    bool enpassant=
                        t2==p.ep;

                    if(capture || enpassant) {

                        if(nr==0 || nr==7) {

                            addMove(
                                moves,s,t2,
                                w?WQ:BQ,false,enpassant
                            );

                            addMove(
                                moves,s,t2,
                                w?WR:BR,false,enpassant
                            );

                            addMove(
                                moves,s,t2,
                                w?WB:BB,false,enpassant
                            );

                            addMove(
                                moves,s,t2,
                                w?WN:BN,false,enpassant
                            );

                        } else {

                            addMove(
                                moves,
                                s,
                                t2,
                                0,
                                false,
                                enpassant
                            );
                        }
                    }
                }
            }
        }

        /* KNIGHT */

        if(pc==WN || pc==BN) {

            const int dr[8]={
                2,2,1,1,-1,-1,-2,-2
            };

            const int df[8]={
                1,-1,2,-2,2,-2,1,-1
            };

            for(int i=0;i<8;i++) {

                int nr=r+dr[i];
                int nf=f+df[i];

                if(nr<0 || nr>7 ||
                   nf<0 || nf>7)
                    continue;

                int t=nr*8+nf;

                if(!ownPiece(p.b[t],w))
                    addMove(moves,s,t);
            }
        }

        /* BISHOP / ROOK / QUEEN */

        bool bishop=
            pc==WB ||
            pc==BB ||
            pc==WQ ||
            pc==BQ;

        bool rook=
            pc==WR ||
            pc==BR ||
            pc==WQ ||
            pc==BQ;

        if(bishop || rook) {

            const int dirs[8][2]={
                {1,1},
                {1,-1},
                {-1,1},
                {-1,-1},
                {1,0},
                {-1,0},
                {0,1},
                {0,-1}
            };

            for(int i=0;i<8;i++) {

                bool diagonal=i<4;

                if(diagonal && !bishop)
                    continue;

                if(!diagonal && !rook)
                    continue;

                int nr=r+dirs[i][0];
                int nf=f+dirs[i][1];

                while(
                    nr>=0 && nr<8 &&
                    nf>=0 && nf<8
                ) {

                    int t=nr*8+nf;

                    if(p.b[t]==EMPTY) {

                        addMove(moves,s,t);

                    } else {

                        if(
                            enemyPiece(p.b[t],w) &&
                            p.b[t]!=(w?BK:WK)
                        ) {
                            addMove(moves,s,t);
                        }

                        break;
                    }

                    nr+=dirs[i][0];
                    nf+=dirs[i][1];
                }
            }
        }

        /* KING */

        if(pc==WK || pc==BK) {

            for(int dr=-1;dr<=1;dr++) {
                for(int df=-1;df<=1;df++) {

                    if(dr==0 && df==0)
                        continue;

                    int nr=r+dr;
                    int nf=f+df;

                    if(nr<0 || nr>7 ||
                       nf<0 || nf>7)
                        continue;

                    int t=nr*8+nf;

                    if(
                        !ownPiece(p.b[t],w) &&
                        p.b[t]!=(w?BK:WK)
                    ) {
                        addMove(moves,s,t);
                    }
                }
            }

            /* CASTLING */

            if(w && s==4 && !inCheck(p,true)) {

                if(
                    p.cwk &&
                    p.b[5]==EMPTY &&
                    p.b[6]==EMPTY &&
                    !squareAttacked(p,5,false) &&
                    !squareAttacked(p,6,false)
                ) {
                    addMove(
                        moves,4,6,0,true,false
                    );
                }

                if(
                    p.cwq &&
                    p.b[1]==EMPTY &&
                    p.b[2]==EMPTY &&
                    p.b[3]==EMPTY &&
                    !squareAttacked(p,3,false) &&
                    !squareAttacked(p,2,false)
                ) {
                    addMove(
                        moves,4,2,0,true,false
                    );
                }
            }

            if(!w && s==60 && !inCheck(p,false)) {

                if(
                    p.cbk &&
                    p.b[61]==EMPTY &&
                    p.b[62]==EMPTY &&
                    !squareAttacked(p,61,true) &&
                    !squareAttacked(p,62,true)
                ) {
                    addMove(
                        moves,60,62,0,true,false
                    );
                }

                if(
                    p.cbq &&
                    p.b[57]==EMPTY &&
                    p.b[58]==EMPTY &&
                    p.b[59]==EMPTY &&
                    !squareAttacked(p,59,true) &&
                    !squareAttacked(p,58,true)
                ) {
                    addMove(
                        moves,60,58,0,true,false
                    );
                }
            }
        }
    }
}

static Position makeMove(
    const Position& p,
    const Move& m
) {
    Position n=p;

    bool w=p.white;

    int pc=n.b[m.from];
    int captured=n.b[m.to];

    n.b[m.from]=EMPTY;

    /* en passant */

    if(m.ep) {

        int cs=m.to+(w?-8:8);

        captured=n.b[cs];
        n.b[cs]=EMPTY;
    }

    n.b[m.to]=
        m.promo ? m.promo : pc;

    /* castling rights */

    if(pc==WK) {
        n.cwk=false;
        n.cwq=false;
    }

    if(pc==BK) {
        n.cbk=false;
        n.cbq=false;
    }

    if(pc==WR && m.from==0)
        n.cwq=false;

    if(pc==WR && m.from==7)
        n.cwk=false;

    if(pc==BR && m.from==56)
        n.cbq=false;

    if(pc==BR && m.from==63)
        n.cbk=false;

    if(captured==WR && m.to==0)
        n.cwq=false;

    if(captured==WR && m.to==7)
        n.cwk=false;

    if(captured==BR && m.to==56)
        n.cbq=false;

    if(captured==BR && m.to==63)
        n.cbk=false;

    /* rook movement during castling */

    if(m.castle) {

        if(w && m.to==6) {
            n.b[5]=WR;
            n.b[7]=EMPTY;
        }

        if(w && m.to==2) {
            n.b[3]=WR;
            n.b[0]=EMPTY;
        }

        if(!w && m.to==62) {
            n.b[61]=BR;
            n.b[63]=EMPTY;
        }

        if(!w && m.to==58) {
            n.b[59]=BR;
            n.b[56]=EMPTY;
        }
    }

    /* en passant target */

    n.ep=-1;

    if(
        (pc==WP || pc==BP) &&
        abs(m.to-m.from)==16
    ) {
        n.ep=(m.to+m.from)/2;
    }

    n.white=!w;

    return n;
}

static vector<Move> legalMoves(
    const Position& p
) {
    vector<Move> pseudo;
    vector<Move> legal;

    generatePseudo(p,pseudo);

    for(const Move& m:pseudo) {

        Position n=makeMove(p,m);

        if(!inCheck(n,p.white))
            legal.push_back(m);
    }

    return legal;
}

/* FEN */

static bool loadFEN(
    Position& p,
    const string& fen
) {
    stringstream ss(fen);

    string board;
    string side;
    string castle;
    string ep;

    if(!(ss>>board>>side>>castle>>ep))
        return false;

    p.b.fill(EMPTY);

    int sq=56;

    for(char c:board) {

        if(c=='/') {
            sq-=16;
            continue;
        }

        if(c>='1' && c<='8') {
            sq+=c-'0';
            continue;
        }

        int pc=EMPTY;

        switch(c) {
            case 'P':pc=WP;break;
            case 'N':pc=WN;break;
            case 'B':pc=WB;break;
            case 'R':pc=WR;break;
            case 'Q':pc=WQ;break;
            case 'K':pc=WK;break;

            case 'p':pc=BP;break;
            case 'n':pc=BN;break;
            case 'b':pc=BB;break;
            case 'r':pc=BR;break;
            case 'q':pc=BQ;break;
            case 'k':pc=BK;break;
        }

        if(pc!=EMPTY) {
            if(sq<0 || sq>=64)
                return false;

            p.b[sq]=pc;
            sq++;
        }
    }

    p.white=(side=="w");

    p.cwk=false;
    p.cwq=false;
    p.cbk=false;
    p.cbq=false;

    for(char c:castle) {

        if(c=='K') p.cwk=true;
        if(c=='Q') p.cwq=true;
        if(c=='k') p.cbk=true;
        if(c=='q') p.cbq=true;
    }

    p.ep=-1;

    if(ep!="-" && ep.size()==2) {
        int f=ep[0]-'a';
        int r=ep[1]-'1';

        if(f>=0 && f<8 &&
           r>=0 && r<8) {
            p.ep=r*8+f;
        }
    }

    return true;
}

/* evaluation */

static int evaluate(
    const Position& p
) {
    int score=0;

    static const int pawnTable[8]={
        0,5,5,10,10,5,5,0
    };

    for(int s=0;s<64;s++) {

        int pc=p.b[s];

        if(pc==EMPTY)
            continue;

        int v=pieceValue(pc);

        if(isWhite(pc)) {

            score+=v;

            if(pc==WP)
                score+=pawnTable[rankOf(s)];

            if(pc==WN || pc==WB)
                score+=rankOf(s)*3;

        } else {

            score-=v;

            int rr=7-rankOf(s);

            if(pc==BP)
                score-=pawnTable[rr];

            if(pc==BN || pc==BB)
                score-=rr*3;
        }
    }

    return p.white ? score : -score;
}

struct SearchInfo {
    chrono::steady_clock::time_point deadline;
    bool stopped=false;
};

static bool timeUp(
    SearchInfo& info
) {
    if(
        chrono::steady_clock::now() >=
        info.deadline
    ) {
        info.stopped=true;
        return true;
    }

    return false;
}

static int search(
    const Position& p,
    int depth,
    int alpha,
    int beta,
    SearchInfo& info
) {
    if(timeUp(info))
        return 0;

    vector<Move> moves=
        legalMoves(p);

    if(moves.empty()) {

        if(inCheck(p,p.white))
            return -MATE+depth;

        return 0;
    }

    if(depth<=0)
        return evaluate(p);

    sort(
        moves.begin(),
        moves.end(),
        [&](const Move& a,const Move& b) {

            int ca=p.b[a.to];
            int cb=p.b[b.to];

            int sa=
                pieceValue(ca)+
                (a.promo?pieceValue(a.promo):0);

            int sb=
                pieceValue(cb)+
                (b.promo?pieceValue(b.promo):0);

            return sa>sb;
        }
    );

    int best=-INF;

    for(const Move& m:moves) {

        int score=
            -search(
                makeMove(p,m),
                depth-1,
                -beta,
                -alpha,
                info
            );

        if(info.stopped)
            return 0;

        if(score>best)
            best=score;

        if(score>alpha)
            alpha=score;

        if(alpha>=beta)
            break;
    }

    return best;
}

static Move findBestMove(
    const Position& p,
    int maxDepth,
    int milliseconds
) {
    vector<Move> moves=
        legalMoves(p);

    Move best=moves.front();

    SearchInfo info;

    info.deadline=
        chrono::steady_clock::now()+
        chrono::milliseconds(milliseconds);

    for(int depth=1;
        depth<=maxDepth;
        depth++) {

        if(timeUp(info))
            break;

        int bestScore=-INF;
        Move depthBest=best;

        for(const Move& m:moves) {

            int score=
                -search(
                    makeMove(p,m),
                    depth-1,
                    -INF,
                    INF,
                    info
                );

            if(info.stopped)
                break;

            if(score>bestScore) {
                bestScore=score;
                depthBest=m;
            }
        }

        if(info.stopped)
            break;

        best=depthBest;
    }

    return best;
}

static string squareName(
    int s
) {
    string x;

    x+=char('a'+fileOf(s));
    x+=char('1'+rankOf(s));

    return x;
}

static string moveToUci(
    const Move& m
) {
    string x=
        squareName(m.from)+
        squareName(m.to);

    if(m.promo) {

        char c='q';

        if(m.promo==WN ||
           m.promo==BN)
            c='n';

        if(m.promo==WB ||
           m.promo==BB)
            c='b';

        if(m.promo==WR ||
           m.promo==BR)
            c='r';

        x+=c;
    }

    return x;
}

static void applyUciMove(
    Position& p,
    const string& text
) {
    if(text.size()<4)
        return;

    int from=
        squareOf(
            text[0],
            text[1]
        );

    int to=
        squareOf(
            text[2],
            text[3]
        );

    int promo=0;

    if(text.size()>=5) {

        char c=text[4];

        if(c=='q')
            promo=p.white?WQ:BQ;

        else if(c=='r')
            promo=p.white?WR:BR;

        else if(c=='b')
            promo=p.white?WB:BB;

        else if(c=='n')
            promo=p.white?WN:BN;
    }

    vector<Move> moves=
        legalMoves(p);

    for(const Move& m:moves) {

        if(
            m.from==from &&
            m.to==to &&
            m.promo==promo
        ) {
            p=makeMove(p,m);
            return;
        }
    }
}

static int parseTime(
    const string& line,
    const string& key,
    int fallback
) {
    size_t pos=line.find(key);

    if(pos==string::npos)
        return fallback;

    pos+=key.size();

    while(
        pos<line.size() &&
        line[pos]==' '
    )
        pos++;

    string n;

    while(
        pos<line.size() &&
        line[pos]>='0' &&
        line[pos]<='9'
    ) {
        n+=line[pos++];
    }

    if(n.empty())
        return fallback;

    return atoi(n.c_str());
}

int main() {

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Position pos=
        startPosition();

    string line;

    while(getline(cin,line)) {

        if(line=="uci") {

            cout
                <<"id name SimpleLogics\n";

            cout
                <<"id author Danny\n";

            cout
                <<"option name Skill Level "
                <<"type spin default 20 min 1 max 20\n";

            cout
                <<"uciok\n";
        }

        else if(line=="isready") {

            cout<<"readyok\n";
        }

        else if(line=="ucinewgame") {

            pos=startPosition();
        }

        else if(
            line.rfind(
                "position startpos",
                0
            )==0
        ) {

            pos=startPosition();

            size_t x=
                line.find("moves");

            if(x!=string::npos) {

                stringstream ss(
                    line.substr(x+5)
                );

                string ms;

                while(ss>>ms)
                    applyUciMove(pos,ms);
            }
        }

        else if(
            line.rfind(
                "position fen",
                0
            )==0
        ) {

            size_t start=
                string("position fen ").size();

            size_t movesPos=
                line.find(" moves ");

            string fen;

            if(movesPos==string::npos)
                fen=line.substr(start);
            else
                fen=line.substr(
                    start,
                    movesPos-start
                );

            loadFEN(pos,fen);

            if(movesPos!=string::npos) {

                stringstream ss(
                    line.substr(
                        movesPos+7
                    )
                );

                string ms;

                while(ss>>ms)
                    applyUciMove(pos,ms);
            }
        }

        else if(
            line.rfind("go",0)==0
        ) {

            vector<Move> moves=
                legalMoves(pos);

            if(moves.empty()) {

                cout
                    <<"bestmove 0000\n";

            } else {

                int movetime=
                    parseTime(
                        line,
                        "movetime ",
                        1000
                    );

                int wtime=
                    parseTime(
                        line,
                        "wtime ",
                        0
                    );

                int btime=
                    parseTime(
                        line,
                        "btime ",
                        0
                    );

                int timeLimit=movetime;

                if(movetime==1000) {

                    int remaining=
                        pos.white?
                        wtime:btime;

                    if(remaining>0) {

                        timeLimit=
                            max(
                                50,
                                min(
                                    3000,
                                    remaining/30
                                )
                            );
                    }
                }

                timeLimit=
                    max(
                        50,
                        min(
                            timeLimit,
                            5000
                        )
                    );

                Move best=
                    findBestMove(
                        pos,
                        8,
                        timeLimit
                    );

                cout
                    <<"bestmove "
                    <<moveToUci(best)
                    <<"\n";
            }
        }

        else if(line=="quit") {
            break;
        }

        cout.flush();
    }

    return 0;
}
