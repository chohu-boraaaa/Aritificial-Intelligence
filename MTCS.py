from abc import ABC, abstractmethod # 추상 클래스 관련 모듈
from collections import defaultdict # 딕셔너리 관련 모듈
import math # 제곱근 계산 위한 모듈

class MCTS: # 몬테카를로 트리 탐색 클래스. 먼저 트리를 롤아웃한 다음 이동을 선택한다.

    def __init__(self, c=1): # 초기화 위한 함수 정의
        self.Q = defaultdict(int)  # 각 노드의 이긴 횟수 값을 0으로 초기화
        self.N = defaultdict(int)  # 각 노드의 방문 횟수 값을 0으로 초기화
        self.children = dict()  # 각 노드의 자식 노드를 딕셔너리 형태로 저장
        self.c = c # 이미 해본 게임에서의 승률을 반영하는 정도와 방문횟수가 작은 노드를 더 탐색해보도록 하는 기회의 반영 정도를 조정하는 역할 상수 c를 의미

    def choose(self, node): # 각 노드에서 최적의 자식 노드를 선택하는 함수 정의
        "최적의 자식 노드를 선택하기."
        if node.is_terminal(): # 만약 단말 노드 상태라면
            raise RuntimeError(f"choose called on terminal node {node}") # 단말 노트 상태라고 오류 발생시키기
        if node not in self.children: # 만약 노드가 자식노드에 포함되지 않는다면
            return node.find_random_child() # 자식노드 무작위 선택
        def score(n): # 각 노드의 승률 구하는 함수 정의
            if self.N[n] ==0: # 만약 한 번도 방문하지 않은 노드라면
                return float("-inf")  # 선택을 하지 않는다
            return self.Q[n] /self.N[n]  # (이긴횟수/게임수(방문수)), 즉 n을 경유한 게임의 승률 반환
        return max(self.children[node], key=score) # 현재까지의 승률(지금까지 이긴 횟수/전체 방문의 수)이 큰 것을 반환

    def do_rollout(self, node): # 트리의 한 층을 더 보여주는 함수 정의
        "트리를 한 층 더 잘 만들기. (한 번 반복할 수 있도록 훈련하기.)"
        path =self._select(node) # 한번도 탐색해보지 않은 자식 노드를 찾아 path에 저장
        leaf = path[-1] # path의 마지막 요소 값을 leaf에 저장
        self._expand(leaf) # leaf의 자식노드 추가하여 확장
        reward =self._simulate(leaf) # leaf의 무작위로 시뮬레이션을 수행한 결과 reward에 저장
        self._backpropagate(path, reward) # 시뮬레이션 승패 정보를 현재 노드에서 루트 노드까지 경로 상에 있는 노드들의 승률 정보에 반영

    def _select(self, node): # 선택 단계 정의
        "탐색해보지 않은 자식노드 찾기"
        path = [] # 방문하는 노드의 경로 저장
        while True: # 참일 동안 반복
            path.append(node) # path에 현재 노드 추가
            if node not in self.children or not self.children[node]: # 만약 현재노드의 자식이거나 후손 노드가 아닐 경우, 즉 아직 탐색해보지 않은 노드거나 단말 노드일 경우
                return path # 현재 노드의 path 반환
            unexplored =self.children[node] -self.children.keys() # 후손노드와 자식노드 키 값의 차집합이 아직 탐색해보지 않은 노드라고 하자
            if unexplored: # 만약 아직 탐색해보지 않은 게 맞다면
                n = unexplored.pop() # n에 아직 탐색해보지 않은 노드의 경로를 저장하고
                path.append(n) # 그 경로를 path에 추가한다.
                return path # path 반환
            node =self._uct_select(node)  # 한 층 아래로 내려가기

    def _expand(self, node): # 확장 단계 정의
        "children에 node의 자식노드를 추가하여 확장하기"
        if node in self.children: # 만약 현재노드의 자식노드라면
            return  # 이미 확장됨
        self.children[node] = node.find_children() # 선택 가능한 이동들을 노드의 children에 추가

    def _simulate(self, node): # 시뮬레이션 단계 정의
        "노드의 임의 시뮬레이션에 대한 결과를 반환."
        invert_reward =True # invert_reward True로 초기화
        while True:
            if node.is_terminal(): # 단말 노드에 도달한다면
                reward = node.reward() # 승패여부 결정한 결과 reward에 저장
                return 1 - reward if invert_reward else reward # 만약 invert_reward가 참이라면 승패 반전 아니라면 그대로 진행
            node = node.find_random_child() # 선택할 수 있는 노드 중에서 무작위 선택
            invert_reward =not invert_reward # invert_reward 상태 반전

    def _backpropagate(self, path, reward): # 역전파 단계 정의
        "시뮬레이션 승패 정보를 현재노드에서 루트 노드까지 보내기"
        for node in reversed(path): # path의 역순, 즉 그 동안의 경로의 역순으로 시뮬레이션 승패 정보 반영
            self.N[node] +=1 # 방문횟수 1 추가
            self.Q[node] += reward # 승패정보 각 노드에 반영
            reward =1 - reward  # 자신이 1이라면 상대는 0으로, 자신이 0이라면 상대는 1으로 만듦

    def _uct_select(self, node): # UCB 정책 적용하는 선택 단계 정의
        "자식노드를 선택할 때 이미 해본 게임에서의 승률을 반영하는 정도와 방문횟수가 작은 노드를 더 탐색해보도록 하는 기회의 반영정도를 조정하기"
        assert all(n in self.children for n in self.children[node]) # 모든 자식노드들이 이미 확장되었는지 확인
        log_N_vertex = math.log(self.N[node]) # UCB 식에서의 logN 만들기
        def uct(n): # UCB 식 정의
            "우선순위 정하는 UCB 값 계산하는 식"
            return self.Q[n] /self.N[n] +self.c * math.sqrt(log_N_vertex /self.N[n]) # UCB 공식
        return max(self.children[node], key=uct) # UCB 값이 가장 큰 수에 해당하는 노드 반환

class Node(ABC):
    "게임 보드판의 상태 표현하기"
    # 추상메서드 이용하여 표현
    @abstractmethod
    def find_children(self):
        "해당 보드판의 모든 가능한 자식노드 표현"
        return set()
    @abstractmethod
    def find_random_child(self):
        "무작위 자식노드 선택"
        return None
    @abstractmethod
    def is_terminal(self):
        "자식노드가 없다면 True 값 반환"
        return True
    @abstractmethod
    def reward(self):
        "승패에 대한 점수 계산, 참고로 1=승, 0=패, 0.5=비김 을 나타내기로 함 ."
        return 0
    @abstractmethod
    def __hash__(self):
        "노드들은 해시적용이 가능해야 함"
        return 123456789
    @abstractmethod
    def __eq__(node1, node2):
        "노드들은 비교가 가능해야 함"
        return True