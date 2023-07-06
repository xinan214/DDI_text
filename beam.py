# 这个类是用于实现束搜索（Beam Search）算法的一个辅助类。
# 束搜索是一种用于在解码过程中生成候选序列的算法，它通过维护一个有序的候选序列集合（beam）来进行搜索。
# 具体来说，这个类用于管理束搜索的状态、更新候选序列和选择最佳解码结果。
class Beam(object):
    """Ordered beam of candidate outputs."""

    # init_symbols 是初始的候选符号（即初始的解码结果）
    def __init__(self, size, init_symbols, cuda=False):
        """Initialize params."""
        self.size = size # 指定束的大小（即候选序列的数量）
        self.done = False
        self.cuda = cuda
        self.scores = torch.FloatTensor(size).zero_()
        if self.cuda:
            self.scores = self.scores.cuda()

        # prevKs和nextYs分别是存储前一步选择的候选序列索引和当前步选择的候选符号
        # previous pointer
        self.prevKs = []
        
        # next step
        self.nextYs = [init_symbols]
   
# get_current_state 方法：返回当前步的候选符号
    def get_current_state(self):
        return self.nextYs[-1]

# get_current_origin 方法：返回当前步的候选序列索引
    def get_current_origin(self):
        return self.prevKs[-1]

    """ Advance the beam """
# advance 方法：根据当前步的候选符号的概率分布，进行一步束搜索的推进。
# workd_lk 是当前步的候选符号的概率分布，大小为 (beam_size, 1, vocab_size)。根据之前的得分（scores）和候选符号的概率分布，计算新的候选序列的得分，并选择得分最高的 beam_size 个候选序列。
# 更新得分（scores）、前一步选择的候选序列索引（prevKs）和当前步选择的候选符号（nextYs）。如果当前步的候选符号中包含终止符（如 EOS_TOKEN），则将 done 设置为 True，表示搜索结束
    def advance(self, workd_lk):
        num_words = workd_lk.size(2)
        
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]
        
        flat_beam_lk = beam_lk.view(-1) 
        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores  
    
        prev_k = bestScoresId / num_words 
        self.prevKs.append(prev_k) #
        self.nextYs.append(bestScoresId - prev_k * num_words) 
        
        # terminal 
        if self.nextYs[-1][0] == EOS_TOKEN:
            self.done = True
        return self.done

# sort_best 方法：对当前步的得分进行排序，返回得分和对应的索引
    def sort_best(self):
        return torch.sort(self.scores, 0, True)

# get_best 方法：返回排序后得分最高的候选序列的得分和索引
    def get_best(self):
        scores, ids = self.sort_best()
        return scores[1], ids[1]

# get_hyp 方法：根据前一步选择的候选序列索引，从最后一步开始回溯，获取最佳解码结果。将解码结果反转后返回
    def get_hyp(self, k):
        hyp = []
        for j in range(len(self.prevKs)-1, -1, -1):
            hyp.append(self.nextYs[j + 1][k].detach().item())
            k = self.prevKs[j][k]
        return hyp[::-1]
    