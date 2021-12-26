use std::cell::RefCell;
use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs;
use std::u8;

fn day01() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let ans = fs::read_to_string("src/input/day01.txt")?
        .trim()
        .split("\n")
        .collect::<Vec<&str>>()
        .iter()
        .map(|&s| s.parse::<u64>().unwrap())
        .collect::<Vec<u64>>()
        .windows(2)
        .filter(|&x| x[1] > x[0])
        .count();
    println!("{:?}", ans);
    Ok(())
}

fn day02() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut x: i64 = 0;
    let mut y: i64 = 0;
    fs::read_to_string("src/input/day02.txt")?
        .trim()
        .split("\n")
        .collect::<Vec<&str>>()
        .iter()
        .for_each(|&s| {
            let v = s.split(" ").nth(1).unwrap().parse::<i64>().unwrap();
            if s.starts_with("forward") {
                x += v
            } else {
                y += if s.starts_with("down") { v } else { -v };
            }
        });
    println!("{:?}", x * y);
    Ok(())
}

fn day03() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut bit_sum = vec![0; 64];
    let mut col_num = 0;
    let mut line_num = 0;
    fs::read_to_string("src/input/day03.txt")?
        .trim()
        .split("\n")
        .collect::<Vec<&str>>()
        .iter()
        .for_each(|&s| {
            for c in s.split("").filter(|&x| x.len() != 0).enumerate() {
                bit_sum[c.0] += c.1.parse::<u64>().unwrap();
                col_num = c.0 + 1;
            }
            line_num += 1;
        });
    bit_sum.truncate(col_num);

    let x: usize = bit_sum
        .iter()
        .map(|&x| if x > line_num / 2 { 1 } else { 0 })
        .zip((0..col_num).rev())
        .map(|(x, i)| x << i)
        .sum();
    println!("{:?}", x * (((1 << col_num) - 1) & !x));
    Ok(())
}

fn day04() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let content = fs::read_to_string("src/input/day04.txt")?;
    let nums = content.trim().split("\n\n").collect::<Vec<&str>>();
    let x: Vec<i32> = nums[0]
        .split(",")
        .collect::<Vec<&str>>()
        .iter()
        .map(|&x| x.parse().unwrap())
        .collect();

    let mut h = HashMap::<i32, i32>::new();
    for (i, v) in x.iter().enumerate() {
        h.insert(*v, i as i32);
    }

    let boards: Vec<Vec<Vec<i32>>> = nums[1..]
        .to_vec()
        .iter()
        .map(|&x| {
            x.split("\n")
                .collect::<Vec<&str>>()
                .iter()
                .map(|&s| {
                    s.split(" ")
                        .collect::<Vec<&str>>()
                        .iter()
                        .filter(|&i| i.len() != 0)
                        .map(|&i| i.parse().unwrap())
                        .collect()
                })
                .collect()
        })
        .collect();

    let (col_num, board_num) = (boards[0][0].len(), boards.len());

    // by row
    let mut rows: Vec<(i32, usize)> = Vec::new();
    boards.iter().zip(0..board_num).for_each(|(b, i)| {
        rows.push((
            *b.iter()
                .map(|r| r.iter().map(|v| h[v]).collect::<Vec<i32>>())
                .collect::<Vec<Vec<i32>>>()
                .iter()
                .map(|r| *r.iter().max().unwrap())
                .collect::<Vec<i32>>()
                .iter()
                .min()
                .unwrap(),
            i,
        ));
    });
    let (index_by_row, board_by_row) = *rows.iter().min_by(|x, y| x.0.cmp(&y.0)).unwrap();

    // by col
    let mut cols: Vec<(i32, usize)> = Vec::new();
    (0..col_num).for_each(|c| {
        let col = boards
            .iter()
            .map(|b| {
                *b.iter()
                    .map(|r| r.iter().map(|v| h[v]).collect::<Vec<i32>>())
                    .collect::<Vec<Vec<i32>>>()
                    .iter()
                    .map(|r| *r.iter().nth(c).unwrap())
                    .collect::<Vec<i32>>()
                    .iter()
                    .max()
                    .unwrap()
            })
            .collect::<Vec<i32>>();
        cols.push((
            *col.iter().min().unwrap(),
            col.iter()
                .position(|v| v == col.iter().min().unwrap())
                .unwrap(),
        ));
    });
    let (index_by_col, board_by_col) = *cols.iter().min_by(|x, y| x.0.cmp(&y.0)).unwrap();

    // choose row or col by min index
    let (index, board) = if index_by_row < index_by_col {
        (index_by_row as usize, board_by_row)
    } else {
        (index_by_col as usize, board_by_col)
    };

    // calc it based on chosen board
    let mut nums: Vec<i32> = Vec::new();
    boards[board].iter().for_each(|x| nums.extend(x.iter()));
    let valid_nums = x[0..index + 1].to_vec();
    let ans = nums.iter().fold(0, |acc, n| {
        if valid_nums.contains(n) {
            return acc;
        }
        acc + n * valid_nums[index]
    });

    println!("{:?}", ans);
    Ok(())
}

fn day05() -> Result<(), Box<dyn std::error::Error + 'static>> {
    #[derive(Copy, Debug)]
    struct Point {
        x: u32,
        y: u32,
    };
    impl Clone for Point {
        fn clone(&self) -> Self {
            *self
        }
    }
    #[derive(Copy, Debug)]
    struct Line {
        start: Point,
        end: Point,
    };
    impl Clone for Line {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl Line {
        fn new(start: Point, end: Point) -> Self {
            Line { start, end }
        }
    }

    let content = fs::read_to_string("src/input/day05.txt")?;
    let mut width: u32 = 0;
    let mut height: u32 = 0;
    let lines: Vec<Line> = content
        .trim()
        .split("\n")
        .map(|r| {
            let ps: Vec<Point> = r
                .trim()
                .split("->")
                .map(|p| {
                    let v: Vec<u32> = p
                        .trim()
                        .split(",")
                        .map(|v| v.parse::<u32>().unwrap())
                        .collect();
                    width = if v[0] > width { v[0] } else { width };
                    height = if v[1] > height { v[1] } else { height };
                    Point { x: v[0], y: v[1] }
                })
                .collect();
            Line::new(ps[0], ps[1])
        })
        .collect();

    let mut counter = vec![vec![0; width as usize + 1]; height as usize + 1];
    lines.iter().for_each(|l| {
        if l.start.x == l.end.x {
            for i in if l.start.y < l.end.y {
                l.start.y..l.end.y + 1
            } else {
                l.end.y..l.start.y + 1
            } {
                counter[i as usize][l.start.x as usize] += 1;
            }
        } else if l.start.y == l.end.y {
            for j in if l.start.x < l.end.x {
                l.start.x..l.end.x + 1
            } else {
                l.end.x..l.start.x + 1
            } {
                counter[l.start.y as usize][j as usize] += 1;
            }
        }
    });

    let ans = counter.iter().flatten().filter(|&&c| c >= 2).count();
    println!("{:?}", ans);

    Ok(())
}

fn day06() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut fish: Vec<i32> = fs::read_to_string("src/input/day06.txt")?
        .trim()
        .split(",")
        .map(|x| x.parse().unwrap())
        .collect();
    for _ in 1..81 {
        for j in 0..fish.len() {
            if fish[j] == 0 {
                fish.push(8);
                fish[j] = 6;
            } else {
                fish[j] -= 1;
            }
        }
    }
    println!("{:?}", fish.len());
    Ok(())
}

fn day07() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut pos: Vec<i32> = fs::read_to_string("src/input/day07.txt")?
        .trim()
        .split(",")
        .map(|x| x.parse().unwrap())
        .collect();
    pos.sort();
    let ans = pos
        .iter()
        .map(|x| (x - pos[pos.len() / 2]).abs())
        .fold(0, |acc, n| acc + n);
    println!("{:?}", ans);
    Ok(())
}

fn day08() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let content = fs::read_to_string("src/input/day08.txt")?;
    let digits: Vec<Vec<&str>> = content
        .trim()
        .split("\n")
        .map(|r| r.trim().split(" | ").nth(1).unwrap().split(" ").collect())
        .collect();
    let ans = digits
        .iter()
        .flatten()
        .filter(|w| w.len() == 2 || w.len() == 3 || w.len() == 4 || w.len() == 7)
        .count();
    println!("{:?}", ans);
    Ok(())
}

fn day09() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut hm: Vec<Vec<i32>> = fs::read_to_string("src/input/day09.txt")?
        .trim()
        .split("\n")
        .map(|r| {
            r.trim()
                .split("")
                .map(|x| if x.len() != 0 { x.parse().unwrap() } else { 9 })
                .collect()
        })
        .collect();
    hm.insert(0, vec![9; hm[0].len()]);
    hm.push(vec![9; hm[0].len()]);

    let mut ans: i32 = 0;
    (1..hm.len() - 1).for_each(|i| {
        (1..hm[0].len() - 1).for_each(|j| {
            if hm[i][j] < hm[i - 1][j]
                && hm[i][j] < hm[i + 1][j]
                && hm[i][j] < hm[i][j - 1]
                && hm[i][j] < hm[i][j + 1]
            {
                ans += hm[i][j] + 1;
            }
        })
    });
    println!("{:?}", ans);
    Ok(())
}

fn day10() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let content = fs::read_to_string("src/input/day10.txt")?;
    let lines: Vec<&str> = content.trim().split("\n").collect();

    let scores: HashMap<&str, i32> = [(")", 3), ("]", 57), ("}", 1197), (">", 25137)]
        .iter()
        .cloned()
        .collect();
    let closes: HashMap<&str, &str> = [(")", "("), ("]", "["), ("}", "{"), (">", "<")]
        .iter()
        .cloned()
        .collect();
    let ans = lines
        .iter()
        .map(|s| {
            let mut stack: Vec<&str> = Vec::new();
            for c in s.split("").filter(|c| c.len() != 0) {
                if ")]}>".contains(c) {
                    if stack.is_empty() || stack.last().unwrap() != &closes[c] {
                        return scores[c];
                    }
                    stack.pop();
                } else {
                    stack.push(c);
                }
            }
            0
        })
        .fold(0, |acc, n| acc + n);
    println!("{:?}", ans);
    Ok(())
}

fn day11() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut octs: Vec<Vec<i32>> = fs::read_to_string("src/input/day11.txt")?
        .trim()
        .split("\n")
        .map(|r| {
            r.split("")
                .filter(|c| c.len() != 0)
                .map(|c| c.parse().unwrap())
                .collect()
        })
        .collect();

    let mut ans: usize = 0;
    let (w, h) = (octs[0].len() as i32, octs.len() as i32);
    (1..101).for_each(|_| {
        let mut roots: VecDeque<(i32, i32)> = VecDeque::new();
        let mut flashes: HashSet<(i32, i32)> = HashSet::new();
        (0..h).for_each(|x| {
            (0..w).for_each(|y| {
                let (i, j) = (x as usize, y as usize);
                octs[i][j] += 1;
                if octs[i][j] == 10 {
                    octs[i][j] = 0;
                    roots.push_back((x, y));
                    flashes.insert((x, y));
                }
            })
        });
        // bfs search
        while !roots.is_empty() {
            let (i, j) = roots.pop_front().unwrap();
            for (di, dj) in vec![
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ] {
                let (x, y) = (i + di, j + dj);
                if x < h && x >= 0 && y < w && y >= 0 {
                    let (i, j) = (x as usize, y as usize);
                    if !flashes.contains(&(x, y)) {
                        octs[i][j] += 1;
                    }
                    if octs[i][j] == 10 {
                        octs[i][j] = 0;
                        roots.push_back((x, y));
                        flashes.insert((x, y));
                    }
                }
            }
        }
        ans += flashes.len();
    });
    println!("{:?}", ans);
    Ok(())
}

fn day12() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let content = fs::read_to_string("src/input/day12.txt")?;
    let mut graph: HashMap<&str, RefCell<Vec<&str>>> = HashMap::new();
    content.trim().split("\n").for_each(|r| {
        let e: Vec<&str> = r.split("-").collect();
        if !graph.contains_key(e[0]) {
            graph.insert(e[0], RefCell::new(vec![e[1]]));
        } else {
            graph[e[0]].borrow_mut().push(e[1]);
        }
        if !graph.contains_key(e[1]) {
            graph.insert(e[1], RefCell::new(vec![e[0]]));
        } else {
            graph[e[1]].borrow_mut().push(e[0]);
        }
    });
    fn dfs<'a>(path: &RefCell<Vec<&'a str>>, graph: &HashMap<&str, RefCell<Vec<&'a str>>>) {
        if path.borrow().is_empty() {
            return;
        }
        let cur = *path.borrow().last().unwrap();
        if cur == "end" {
            //println!("{:?}", path);
        } else if graph.contains_key(cur) {
            for next in graph[cur].borrow().iter() {
                if (next.to_lowercase() == *next
                    && path.borrow().iter().position(|n| n == next).is_none())
                    || next.to_uppercase() == *next
                {
                    path.borrow_mut().push(next);
                    dfs(path, graph);
                }
            }
        }
        path.borrow_mut().pop();
    }
    let path: RefCell<Vec<&str>> = RefCell::new(vec!["start"]);
    dfs(&path, &graph);
    Ok(())
}

fn day13() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let content = fs::read_to_string("src/input/day13.txt")?;
    let parts: Vec<&str> = content.trim().split("\n\n").collect();
    let mut w: usize = 0;
    let mut h: usize = 0;
    let dots: Vec<Vec<usize>> = parts[0]
        .trim()
        .split("\n")
        .map(|r| {
            let v: Vec<usize> = r.split(",").map(|n| n.parse().unwrap()).collect();
            w = if v[0] > w { v[0] } else { w };
            h = if v[1] > h { v[1] } else { h };
            v
        })
        .collect();
    let mut board = vec![vec![0; w + 1]; h + 1];
    dots.iter().for_each(|d| board[d[1]][d[0]] = 1);

    #[derive(Debug)]
    enum Op {
        FoldX(usize),
        FoldY(usize),
    }
    let insts: Vec<Op> = parts[1]
        .trim()
        .split("\n")
        .map(|r| {
            let parts: Vec<&str> = r.trim().split("=").collect();
            if parts[0] == "fold along x" {
                Op::FoldX(parts[1].parse().unwrap())
            } else {
                Op::FoldY(parts[1].parse().unwrap())
            }
        })
        .collect();

    insts.iter().for_each(|op| match op {
        Op::FoldX(x) => {
            (0..h + 1).for_each(|i| {
                let (mut l, mut r) = (*x as i32 - 1, x + 1);
                while l >= 0 && r <= w {
                    let j = l as usize;
                    board[i][j] = if (board[i][j] + board[i][r]) >= 1 {
                        1
                    } else {
                        0
                    };
                    board[i][r] = 0;
                    l -= 1;
                    r += 1;
                }
            });
            w = x - 1;
        }
        Op::FoldY(y) => {
            (0..w + 1).for_each(|j| {
                let (mut u, mut b) = (*y as i32 - 1, y + 1);
                while u >= 0 && b <= h {
                    let i = u as usize;
                    board[i][j] = if (board[i][j] + board[b][j]) >= 1 {
                        1
                    } else {
                        0
                    };
                    board[b][j] = 0;
                    u -= 1;
                    b += 1;
                }
            });
            h = y - 1;
        }
    });
    let ans = board.iter().flatten().filter(|&&v| v > 0).count();
    println!("{:?}", ans);
    (0..h + 1).for_each(|i| {
        (0..w + 1).for_each(|j| {
            print!("{}", if board[i][j] == 1 { "###" } else { "..." });
        });
        println!();
    });
    Ok(())
}

fn day14() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let content = fs::read_to_string("src/input/day14.txt")?;
    let parts: Vec<&str> = content.trim().split("\n\n").collect();
    let template = parts[0].trim().to_owned();
    let insts: HashMap<&str, (String, String)> = parts[1]
        .trim()
        .split("\n")
        .map(|r| {
            let v: Vec<&str> = r.trim().split(" -> ").collect();
            (
                v[0],
                (v[0][..1].to_owned() + v[1], v[1].to_owned() + &v[0][1..]),
            )
        })
        .collect();

    fn inc_by(counter: &mut HashMap<String, i64>, key: &str, val: i64) {
        match counter.get_mut(key) {
            Some(v) => {
                *v += val;
            }
            None => {
                counter.insert(key.to_string(), val);
            }
        }
    }
    let mut round: HashMap<String, i64> = HashMap::new();
    template
        .split("")
        .filter(|e| e.len() != 0)
        .collect::<Vec<&str>>()
        .windows(2)
        .map(|w| w.iter().fold("".to_string(), |acc, n| acc + n))
        .collect::<Vec<String>>()
        .iter()
        .for_each(|e| inc_by(&mut round, e, 1));
    (1..41).for_each(|_| {
        let mut next_round: HashMap<String, i64> = HashMap::new();
        for i in round.iter() {
            let (p0, p1) = insts.get(&i.0[..]).unwrap();
            inc_by(&mut next_round, p0, round[i.0]);
            inc_by(&mut next_round, p1, round[i.0]);
        }
        round = next_round;
    });

    let mut counter: HashMap<String, i64> = HashMap::new();
    for e in round.iter() {
        inc_by(&mut counter, &e.0[..1], round[e.0]);
        inc_by(&mut counter, &e.0[1..], round[e.0]);
    }

    fn handle_first_last_char(template: &str, pair: (&String, &i64)) -> i64 {
        let len = template.len();
        let mut ans = *pair.1;
        ans += if &template[..1] == pair.0 { 1 } else { 0 };
        ans += if &template[len - 1..] == pair.0 { 1 } else { 0 };
        ans / 2
    }
    let most = handle_first_last_char(
        &template,
        counter.iter().max_by(|x, y| x.1.cmp(&y.1)).unwrap(),
    );
    let least = handle_first_last_char(
        &template,
        counter.iter().min_by(|x, y| x.1.cmp(&y.1)).unwrap(),
    );

    println!("{:?}", most - least);
    Ok(())
}

fn day15() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let content = fs::read_to_string("src/input/day15.txt")?;
    let board: Vec<Vec<i64>> = content
        .trim()
        .split("\n")
        .map(|r| {
            r.trim()
                .split("")
                .filter(|v| v.len() != 0)
                .map(|v| v.parse().unwrap())
                .collect()
        })
        .collect();
    const TIMES: usize = 5;
    let (w, h) = (board[0].len(), board.len());
    let mut ans = vec![0; w * TIMES];
    fn pos_val(board: &Vec<Vec<i64>>, i: usize, j: usize) -> i64 {
        let (w, h) = (board[0].len(), board.len());
        let val = board[i % h][j % w] + (i / h) as i64 + (j / w) as i64;
        val % 10 + val / 10
    }
    (1..w * TIMES).for_each(|j| {
        ans[j] = ans[j - 1] + pos_val(&board, 0, j);
    });
    (1..h * TIMES).for_each(|i| {
        ans[0] += pos_val(&board, i, 0);
        (1..w * TIMES).for_each(|j| {
            ans[j] = cmp::min(
                ans[j] + pos_val(&board, i, j),
                ans[j - 1] + pos_val(&board, i, j),
            );
        });
    });
    println!("{:?}", ans[w * TIMES - 1]);
    Ok(())
}

fn day16() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let content = fs::read_to_string("src/input/day16.txt")?;
    let bytes: Vec<u8> = content
        .trim()
        .split("")
        .filter(|b| b.len() != 0)
        .map(|b| u8::from_str_radix(b, 16).unwrap())
        .collect();
    // bytes.iter().for_each(|b| print!("{:#06b}", b));

    #[derive(Debug)]
    struct Parser<'a> {
        bytes: &'a Vec<u8>,
        id: usize,
        off: usize,
        ver: u128,
    }

    #[derive(Debug)]
    enum Expr {
        Sum,
        Product,
        Min,
        Max,
        Gt,
        Lt,
        Eq,
        Num(u128),
        List(Vec<Expr>),
    }

    impl<'a> Parser<'a> {
        fn position(&self) -> usize {
            self.id * 4 + self.off
        }

        fn recv_bits(&mut self, n: usize) -> Vec<u8> {
            let mut bytes: Vec<u8>;
            // inclusive bit range [first:first_off, last:last_off]
            let first_bit_id = self.id * 4 + self.off;
            let last_bit_id = first_bit_id + n - 1;
            let (first, first_off) = (self.id, self.off);
            let (last, last_off) = (last_bit_id / 4, last_bit_id % 4);
            // bytes which contains the bit range we need
            bytes = (&self.bytes[first..last + 1]).to_vec();
            // remove leading out of range bits
            bytes[0] &= (1 << (4 - first_off)) - 1;
            // bytewise shift bits backward
            let mut i = bytes.len() - 1;
            while last_off != 3 && i >= 1 {
                bytes[i] = ((bytes[i - 1] & ((1 << (4 - last_off - 1)) - 1)) << (last_off + 1))
                    + (bytes[i] >> (4 - last_off - 1));
                i -= 1;
            }
            bytes[0] >>= 4 - last_off - 1;
            // update inner state to next unparsed bit's position
            self.id = last + (last_off + 1) / 4;
            self.off = (last_off + 1) % 4;
            bytes
        }
        fn bytes_to_num(bytes: &Vec<u8>) -> u128 {
            bytes
                .iter()
                .zip((0..bytes.len()).rev())
                .map(|(x, i)| (*x as u128) << 4 * i)
                .sum()
        }
        fn recv_bytes_as_num(&mut self, num: usize) -> u128 {
            let bytes = self.recv_bits(num);
            Parser::bytes_to_num(&bytes)
        }
        fn recv_ver(&mut self) -> u128 {
            self.recv_bytes_as_num(3)
        }
        fn recv_typ(&mut self) -> u128 {
            self.recv_bytes_as_num(3)
        }
        fn recv_len_typ(&mut self) -> u128 {
            self.recv_bytes_as_num(1)
        }
        fn recv_len(&mut self) -> u128 {
            self.recv_bytes_as_num(15)
        }
        fn recv_num(&mut self) -> u128 {
            self.recv_bytes_as_num(11)
        }

        fn parse_literal(&mut self) -> Expr {
            let mut val_bytes: Vec<u8> = Vec::new();
            loop {
                let is_last_group = self.recv_bits(1)[0] == 0;
                let bytes = self.recv_bits(4);
                val_bytes.push(Parser::bytes_to_num(&bytes) as u8);
                if is_last_group {
                    break;
                }
            }
            Expr::Num(Parser::bytes_to_num(&val_bytes))
        }

        pub fn parse_packet(&mut self) -> Expr {
            let ver = self.recv_ver();
            let typ = self.recv_typ();
            self.ver += ver;
            if typ == 4 {
                // literal number
                self.parse_literal()
            } else {
                // sub-packets
                let mut exprs: Vec<Expr> = Vec::new();
                let op = match typ {
                    0 => Expr::Sum,
                    1 => Expr::Product,
                    2 => Expr::Min,
                    3 => Expr::Max,
                    5 => Expr::Gt,
                    6 => Expr::Lt,
                    7 => Expr::Eq,
                    _ => unreachable!(),
                };
                exprs.push(op);

                let len_typ = self.recv_len_typ();
                if len_typ == 0 {
                    let len = self.recv_len();
                    let start = self.position();
                    loop {
                        if self.position() - start >= len as usize {
                            break;
                        }
                        exprs.push(self.parse_packet());
                    }
                } else {
                    let num = self.recv_num();
                    for _ in 0..num {
                        exprs.push(self.parse_packet());
                    }
                }

                Expr::List(exprs)
            }
        }
    }

    impl Expr {
        fn eval(&self) -> u128 {
            use Expr::*;
            match self {
                Num(v) => *v,
                List(v) => {
                    let op = &v[0];
                    let mut nums: Vec<u128> = Vec::new();
                    for e in &v[1..] {
                        nums.push(e.eval());
                    }
                    let v = match op {
                        Sum => nums.iter().fold(0, |acc, n| acc + n),
                        Product => nums.iter().fold(1, |acc, n| acc * n),
                        Min => *nums.iter().min().unwrap(),
                        Max => *nums.iter().max().unwrap(),
                        Gt => {
                            let v = if nums[0] > nums[1] { 1 } else { 0 };
                            v
                        }
                        Lt => {
                            let v = if nums[0] < nums[1] { 1 } else { 0 };
                            v
                        }
                        Eq => {
                            let v = if nums[0] == nums[1] { 1 } else { 0 };
                            v
                        }
                        _ => unreachable!(),
                    };
                    v
                }
                _ => unreachable!(),
            }
        }
    }

    let mut parser = Parser {
        bytes: &bytes,
        id: 0,
        off: 0,
        ver: 0,
    };
    let expr = parser.parse_packet();
    println!("{} {}", parser.ver, expr.eval());
    Ok(())
}

fn main() {
    assert!(day01().is_ok());
    assert!(day02().is_ok());
    assert!(day03().is_ok());
    assert!(day04().is_ok());
    assert!(day05().is_ok());
    assert!(day06().is_ok());
    assert!(day07().is_ok());
    assert!(day08().is_ok());
    assert!(day09().is_ok());
    assert!(day10().is_ok());
    assert!(day11().is_ok());
    assert!(day12().is_ok());
    assert!(day13().is_ok());
    assert!(day14().is_ok());
    assert!(day15().is_ok());
    assert!(day16().is_ok());
}
