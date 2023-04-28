import numpy as np
from gym_2048.envs import TwentyFortyeight

def test_reset():
    env = TwentyFortyeight()
    obs, _ = env.reset()
    assert obs.shape == (4,4)
    assert len(np.where(obs == 0)[0]) == 14
    assert len(np.where(obs == 0)[1]) == 14

def test_get_num_spawn():
    env = TwentyFortyeight()
    for x in range(4):
        for y in range(4):
            env._spawn(x, y, x*4+y)
    
    for x in range(4):
        for y in range(4):
            assert env._get_num(x, y) == x*4+y

def test_gameover():
    env = TwentyFortyeight()
    """
     1  2  3  4       1  2  3  4       1  2  3  4 
     5  6  7  8  -->  5  6  7  8  -->  5  6  7  8
     9 10 11 12 right 9 10 11 12 spawn 9 10 11 12
    13 14 15 15       0 13 14 16       1 13 14 16
    """
    for x in range(4):
        for y in range(4):
            if x == 3 and y == 3:
                env._spawn(x, y, 15)
            else:
                env._spawn(x, y, x*4+y+1)
    env._set_legal_actions()
    o, r, done, t, _ = env.step(0)
    assert done

def test_right():
    env = TwentyFortyeight()
    """
    0 0 1 1     0 0 0 2
    3 3 4 4 --> 0 0 4 5
    4 2 2 3     0 4 3 3
    1 2 3 4     1 2 3 4
    """
    env._spawn(0, 0, 0); env._spawn(0, 1, 0); env._spawn(0, 2, 1); env._spawn(0, 3, 1)
    env._spawn(1, 0, 3); env._spawn(1, 1, 3); env._spawn(1, 2, 4); env._spawn(1, 3, 4)
    env._spawn(2, 0, 4); env._spawn(2, 1, 2); env._spawn(2, 2, 2); env._spawn(2, 3, 3)
    env._spawn(3, 0, 1); env._spawn(3, 1, 2); env._spawn(3, 2, 3); env._spawn(3, 3, 4)
    o, r, d, t, _ = env.step(0)
    assert(r == 2**2+2**4+2**5+2**3)
    assert(env._get_num(0, 0)==0); assert(env._get_num(0, 1)==0); assert(env._get_num(0, 2)==0); assert(env._get_num(0, 3)==2) 
    assert(env._get_num(1, 0)==0); assert(env._get_num(1, 1)==0); assert(env._get_num(1, 2)==4); assert(env._get_num(1, 3)==5) 
    assert(env._get_num(2, 0)==0); assert(env._get_num(2, 1)==4); assert(env._get_num(2, 2)==3); assert(env._get_num(2, 3)==3) 
    assert(env._get_num(3, 0)==1); assert(env._get_num(3, 1)==2); assert(env._get_num(3, 2)==3); assert(env._get_num(3, 3)==4) 

def test_down():
    env = TwentyFortyeight()
    """
    0 4 1 1     0 0 0 1
    0 4 6 2 --> 0 0 1 2
    2 5 6 3     0 5 7 3
    2 5 7 4     3 6 7 4
    """
    env._spawn(0, 0, 0); env._spawn(0, 1, 4); env._spawn(0, 2, 1); env._spawn(0, 3, 1)
    env._spawn(1, 0, 0); env._spawn(1, 1, 4); env._spawn(1, 2, 6); env._spawn(1, 3, 2)
    env._spawn(2, 0, 2); env._spawn(2, 1, 5); env._spawn(2, 2, 6); env._spawn(2, 3, 3)
    env._spawn(3, 0, 2); env._spawn(3, 1, 5); env._spawn(3, 2, 7); env._spawn(3, 3, 4)
    o, r, d, t, _ = env.step(1)
    assert(r == 2**3+2**5+2**6+2**7)
    assert(env._get_num(0, 0)==0); assert(env._get_num(0, 1)==0); assert(env._get_num(0, 2)==0); assert(env._get_num(0, 3)==1) 
    assert(env._get_num(1, 0)==0); assert(env._get_num(1, 1)==0); assert(env._get_num(1, 2)==1); assert(env._get_num(1, 3)==2) 
    assert(env._get_num(2, 0)==0); assert(env._get_num(2, 1)==5); assert(env._get_num(2, 2)==7); assert(env._get_num(2, 3)==3) 
    assert(env._get_num(3, 0)==3); assert(env._get_num(3, 1)==6); assert(env._get_num(3, 2)==7); assert(env._get_num(3, 3)==4) 

def test_left():
    env = TwentyFortyeight()
    """
     0  0 12 12     13  0  0  0
    10 10 11 11 --> 11 12  0  0
    15 14 14 13     15 15 13  0
    16 15 14 13     16 15 14 13
    """
    env._spawn(0, 0,  0); env._spawn(0, 1,  0); env._spawn(0, 2, 12); env._spawn(0, 3, 12)
    env._spawn(1, 0, 10); env._spawn(1, 1, 10); env._spawn(1, 2, 11); env._spawn(1, 3, 11)
    env._spawn(2, 0, 15); env._spawn(2, 1, 14); env._spawn(2, 2, 14); env._spawn(2, 3, 13)
    env._spawn(3, 0, 16); env._spawn(3, 1, 15); env._spawn(3, 2, 14); env._spawn(3, 3, 13)
    o, r, d, t, _ = env.step(2)
    assert(r == 2**13+2**11+2**12+2**15)
    assert(env._get_num(0, 0)==13); assert(env._get_num(0, 1)== 0); assert(env._get_num(0, 2)== 0); assert(env._get_num(0, 3)== 0) 
    assert(env._get_num(1, 0)==11); assert(env._get_num(1, 1)==12); assert(env._get_num(1, 2)== 0); assert(env._get_num(1, 3)== 0) 
    assert(env._get_num(2, 0)==15); assert(env._get_num(2, 1)==15); assert(env._get_num(2, 2)==13); assert(env._get_num(2, 3)== 0) 
    assert(env._get_num(3, 0)==16); assert(env._get_num(3, 1)==15); assert(env._get_num(3, 2)==14); assert(env._get_num(3, 3)==13) 

def test_up():
    env = TwentyFortyeight()
    """
    0 4 1 1     3 5 1 1
    0 4 6 2 --> 0 6 7 2
    2 5 6 3     0 0 7 3
    2 5 7 4     0 0 0 4
    """
    env._spawn(0, 0, 0); env._spawn(0, 1, 4); env._spawn(0, 2, 1); env._spawn(0, 3, 1)
    env._spawn(1, 0, 0); env._spawn(1, 1, 4); env._spawn(1, 2, 6); env._spawn(1, 3, 2)
    env._spawn(2, 0, 2); env._spawn(2, 1, 5); env._spawn(2, 2, 6); env._spawn(2, 3, 3)
    env._spawn(3, 0, 2); env._spawn(3, 1, 5); env._spawn(3, 2, 7); env._spawn(3, 3, 4)
    o, r, d, t, _ = env.step(3)
    assert(r == 2**3+2**5+2**6+2**7)
    assert(env._get_num(0, 0)==3); assert(env._get_num(0, 1)==5); assert(env._get_num(0, 2)==1); assert(env._get_num(0, 3)==1) 
    assert(env._get_num(1, 0)==0); assert(env._get_num(1, 1)==6); assert(env._get_num(1, 2)==7); assert(env._get_num(1, 3)==2) 
    assert(env._get_num(2, 0)==0); assert(env._get_num(2, 1)==0); assert(env._get_num(2, 2)==7); assert(env._get_num(2, 3)==3) 
    assert(env._get_num(3, 0)==0); assert(env._get_num(3, 1)==0); assert(env._get_num(3, 2)==0); assert(env._get_num(3, 3)==4) 
