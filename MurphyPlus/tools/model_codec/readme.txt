1.在生成bin文件的时候，只把权重存在bin文件里边即可
2.解析protxt文件，录入层

p1 {
	item1:val1
	item2:val2
	p2{
		item3:val230
		item3:val231
	}
	p2 {
		p3 {
			item4:val4
		}
	}
}
第一个中括号：[parent_title:parent_idx same_parent_num]
第二个中括号：[item_idx]
item1:[p1:0 1][0]
item2:[p1:0 1][0]
item3:[p1:0 1] []
