'''
用 python 实现简单的链表增删遍查操作
'''
# 生成节点
class Node(object):
    def __init__(self,data):
        self.val = data
        self.next = None

    # 得到当前节点的值
    def get_Data(self):
        return self.val

    # 设置节点数据
    def set_Data(self,new_data):
        self.val = new_data

    # 返回后继节点
    def get_Next(self):
        return self.next

    # 更新后继节点
    def set_Next(self,new_next):
        self.next = new_next

# 对于链表的操作
class LinkList(object):
    def __init__(self):
        self.head = None

    # 前插法 添加节点使之成为新的头节点
    def add_Node(self,data):
        new_Node = Node(data)
        new_Node.set_Next(self.head)
        self.head = new_Node

    # 查找链表，是否有对应的值
    def searching(self,data):
        checking = self.head      # 从头节点开始
        while checking != None:   # 如果节点不为空
            if checking.get_Data() == data:   # 得到节点内的值判断是否与所给数值相等
                return True
            checking = checking.get_Next()    # 若没有，移动到下一个节点继续判断
        return False

    # 删除节点
    def removeNode(self,data):
        checking = self.head    # 从头节点开始
        previous = None         # 保存要删除节点的前一个节点
        while checking != None:
            if checking.get_Data() == data:
                break
            previous = checking              # previous等于checking更新之前的节点
            checking = checking.get_Next()

        if previous == None:                 # 如果查找到的第一个节点为头节点，直接将头节点的指针指向下一个
            self.head = checking.get_Next()
        else:
            previous.set_Next(checking.get_Next())   # 前一个节点的指针，指向删除节点的下一个节点

    def sizeNode(self):  # 统计节点总数
        size = 0
        temp = self.head
        while temp != None:
            size += 1
            temp = temp.get_Next()
        return size

    def isEmpty(self):
        return self.head == None

tmp = list()

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        '''
        # 先前向遍历列表，将各节点的值保存到一个list，最后list进行反转
        lists = []
        while listNode != None:
            lists.append(listNode.val)
            listNode = listNode.next
        return lists[::-1]
        '''
        if listNode != None:
            if listNode.next != None:
                self.printListFromTailToHead(listNode.next)
            print(listNode.val)
            tmp.append(listNode.val)
            print(tmp)
        return tmp

if __name__ == '__main__':
    l = LinkList()
    for i in range(10):
        l.add_Node(i)
    print(l.sizeNode())
    Solution().printListFromTailToHead(l.head)

