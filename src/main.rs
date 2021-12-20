use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs;

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
        // println!();
        // (0..h).for_each(|i| {
        //     (0..w).for_each(|j| {
        //         if flashes.contains(&(i, j)) {
        //             print!("0 ");
        //         } else {
        //             print!("{:?} ", octs[i as usize][j as usize]);
        //         }
        //     });
        //     println!();
        // });
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
    });
    println!("{:?}", graph);
    let mut stack: Vec<&str> = vec!["start"];
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
}