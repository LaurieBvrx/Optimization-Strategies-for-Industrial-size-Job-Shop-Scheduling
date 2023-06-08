import heapq

def equalListOfList(l1, l2):
    # return True if the two lists of lists are equal
    if len(l1) != len(l2):
        return False
    else:
        for i in range(len(l1)):
            if len(l1[i]) != len(l2[i]):
                return False
            else:
                for j in range(len(l1[i])):
                    if l1[i][j] != l2[i][j]:
                        return False
        return True

def reduceList(l_param):
    l = l_param.copy()
    reduce = True
    while reduce and  len(l) > 1:        
        reduce = l[1] == l[0]
        # if l[0] != 0:
        #     reduce = reduce or l[1] == l[0] + 1
        reduce = reduce and len(l) > 1
        if reduce:
            l.pop(0)
            l[0] = l[0] + 1
    return l

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (priority, _, item) = heapq.heappop(self.heap)
        return (priority, item)

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)