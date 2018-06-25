#ifndef __DLIST_H
#define __DLIST_H

/* This file is from Linux Kernel (include/linux/list.h)
* and modified by simply removing hardware prefetching of list items.
* Here by copyright, credits attributed to wherever they belong.
* Kulesh Shanmugasundaram (kulesh [squiggly] isis.poly.edu)
*/

/*
* Simple doubly linked list implementation.
*
* Some of the internal functions (“__xxx”) are useful when
* manipulating whole lists rather than single entries, as
* sometimes we already know the next/prev entries and we can
* generate better code by using them directly rather than
* using the generic single-entry routines.
*/
/**
 * container_of - cast a member of a structure out to the containing structure
 *
 * @ptr:	the pointer to the member.
 * @type:	the type of the container struct this is embedded in.
 * @member:	the name of the member within the struct.
 *
 */
#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

#define container_of(ptr, type, member) ({			\
        const typeof( ((type *)0)->member ) *__mptr = (ptr);	\
        (type *)( (char *)__mptr - offsetof(type,member) );})

/*
 * These are non-NULL pointers that will result in page faults
 * under normal circumstances, used to verify that nobody uses
 * non-initialized list entries.
 */
// 两个安全的指针指向，也可以让它自己指向自己，防止悬垂
#define LIST_POISON1  ((void *) 0x00100100)
#define LIST_POISON2  ((void *) 0x00200)

struct list_head {
	struct list_head *next, *prev;
};

//初始化结构体
#define LIST_HEAD_INIT(name) { &(name), &(name) }

//使name变量的前向指针和后向指针都指向自己
#define LIST_HEAD(name) \
struct list_head name = LIST_HEAD_INIT(name)

// (对小结构体指针变量的初始化)struct list_head *ptr 
#define INIT_LIST_HEAD(ptr) do { \
	(ptr)->next = (ptr); (ptr)->prev = (ptr); \
} while (0)

/*
* Insert a new_list entry between two known consecutive entries.
*
* This is only for internal list manipulation where we know
* the prev/next entries already!
*/
//添加new到prev和next之间
static inline void __list_add(struct list_head *new_list,
				struct list_head *prev,
				struct list_head *next)
{
	next->prev = new_list;
	new_list->next = next;
	new_list->prev = prev;
	prev->next = new_list;
}

/**
* list_add – add a new_list entry
* @new_list: new_list entry to be added
* @head: list head to add it after
*
* Insert a new_list entry after the specified head.
* This is good for implementing stacks.
*/
//将new节点头插入
static inline void _list_add(struct list_head *new_list, struct list_head *head)
{
	__list_add(new_list, head, head->next);
}

/**
* list_add_tail – add a new_list entry
* @new_list: new_list entry to be added
* @head: list head to add it before
*
* Insert a new_list entry before the specified head.
* This is useful for implementing queues.
*/
//将new节点尾插入（也是插入头部前边因为是循环链表）
static inline void list_add_tail(struct list_head *new_list, struct list_head *head)
{
	__list_add(new_list, head->prev, head);
}


static inline void list_add_after(struct list_head *new_list,struct list_head *p)
{
	p->next->prev = new_list;
	new_list->next = p->next;
	p->next = new_list;
	new_list->prev = p;
}



static inline void list_add_before(struct list_head *new_list,struct list_head *p)
{
	p->prev->next = new_list;
	new_list->prev = p->prev;
	new_list->next = p;
	p->prev = new_list;
}

/*
* Delete a list entry by making the prev/next entries
* point to each other.
*
* This is only for internal list manipulation where we know
* the prev/next entries already!
*/
//删除prev和next之间的结点
static inline void __list_del(struct list_head *prev, struct list_head *next)
{
	next->prev = prev;
	prev->next = next;
}

/**
* list_del – deletes entry from list.
* @entry: the element to delete from the list.
* Note: list_empty on entry does not return true after this, the entry is in an undefined state.
*/
//entry 是将要被删除的结点，linux内核链表不会执行free操作，而是由用户来操作。
static inline void list_del(struct list_head *entry)
{
	__list_del(entry->prev, entry->next);
	entry->next = (struct list_head *) 0;
	entry->prev = (struct list_head *) 0;
}

/**
* list_del_init – deletes entry from list and reinitialize it.
* @entry: the element to delete from the list.
*/
//entry指向的结点被删除之后，使其指针域都指向其本身
static inline void list_del_init(struct list_head *entry)
{
	__list_del(entry->prev, entry->next);
	INIT_LIST_HEAD(entry);
}

/**
* list_move – delete from one list and add as another’s head
* @list: the entry to move
* @head: the head that will precede our entry
*/
//删除list指向的结点，将其插入到head的后边。
static inline void list_move(struct list_head *list,
				struct list_head *head)
{
	__list_del(list->prev, list->next);
	_list_add(list, head);
}

/**
* list_move_tail – delete from one list and add as another’s tail
* @list: the entry to move
* @head: the head that will follow our entry
*/
//删除list指向的结点，将其插入到head的前边
static inline void list_move_tail(struct list_head *list,
					struct list_head *head)
{
	__list_del(list->prev, list->next);
	list_add_tail(list, head);
}

/**
* list_empty – tests whether a list is empty
* @head: the list to test.
*/
//判断一个链表是否为空
static inline int list_empty(struct list_head *head)
{
	return head->next == head;
}
//实现两个链表的合并,将list链表的头结点删除，将其所剩余结点插入到head和at之间
static inline void __list_splice(struct list_head *list,
					struct list_head *head)
{
	struct list_head *first = list->next;
	struct list_head *last = list->prev;
	struct list_head *at = head->next;

	first->prev = head;
	head->next = first;

	last->next = at;
	at->prev = last;
}

/**
* list_splice – join two lists
* @list: the new_list list to add.
* @head: the place to add it in the first list.
*/
static inline void list_splice(struct list_head *list, struct list_head *head)
{
if (!list_empty(list))
__list_splice(list, head);
}

/**
* list_splice_init – join two lists and reinitialise the emptied list.
* @list: the new_list list to add.
* @head: the place to add it in the first list.
*
* The list at @list is reinitialised
*///将list链表的头结点的指针指向其本身
static inline void list_splice_init(struct list_head *list,
struct list_head *head)
{
if (!list_empty(list)) {
__list_splice(list, head);
INIT_LIST_HEAD(list);
}
}

/**
* list_entry – get the struct for this entry
* @ptr:    the &struct list_head pointer.
* @type:    the type of the struct this is embedded in.
* @member:    the name of the list_struct within the struct.
*/
//该宏的实现可以分为两部分，减号左边为小结构体的实际地址，右边为
//是小结构体相对于大结构体的偏移量。
//该宏的返回值是大结构体的地址。
#define list_entry(ptr, type, member) \
((type *)((char *)(ptr)-(unsigned long)(&((type *)0)->member)))

/**
* list_for_each    -    iterate over a list
* @pos:    the &struct list_head to use as a loop counter.
* @head:    the head for your list.
*/
//对小结构体从前向后遍历
#define list_for_each(pos, head) \
for (pos = (head)->next; pos != (head); pos = pos->next)
/**
* list_for_each_prev    -    iterate over a list backwards
* @pos:    the &struct list_head to use as a loop counter.
* @head:    the head for your list.
*/
//对小结构体从后向前遍历
#define list_for_each_prev(pos, head) \
for (pos = (head)->prev; pos != (head); \
pos = pos->prev)

/**
* list_for_each_safe    -    iterate over a list safe against removal of list entry
* @pos:    the &struct list_head to use as a loop counter.
* @n:        another &struct list_head to use as temporary storage
* @head:    the head for your list.
*/
//当遍历当前结点时，记住下一个结点地址，及时当前被遍历的结点意外被删除，
//也不至于产生断链。
#define list_for_each_safe(pos, n, head) \
for (pos = (head)->next, n = pos->next; pos != (head); \
pos = n, n = pos->next)

/**
* list_for_each_entry    -    iterate over list of given type
* @pos:    the type * to use as a loop counter.
* @head:    the head for your list.
* @member:    the name of the list_struct within the struct.
*/
//对大结构体的遍历，根据小结构来遍历大结构体，pos为大结构体地址
/*list_for_each_entry(pos,head,memeber)
	pos   -->为结构体的指针
	head  -->结构体类型
	member-->结构体中的成员 这里指结构体里的list_head

typedef	struct A
{
	省略
}link; 
link pos[n];
pos为结构体指针 memeber为结构体中list_head的两个指针的组合
*/
#define list_for_each_entry(pos, head, member,data_type)                \
for (pos = list_entry((head)->next, data_type, member);    \
&pos->member != (head);                     \
pos = list_entry(pos->member.next, data_type, member))

/**
* list_for_each_entry_safe – iterate over list of given type safe against removal of list entry
* @pos:    the type * to use as a loop counter.
* @n:        another type * to use as temporary storage
* @head:    the head for your list.
* @member:    the name of the list_struct within the struct.
*/
#define list_for_each_entry_safe(pos, n, head, member)            \
for (pos = list_entry((head)->next, typeof(*pos), member),    \
n = list_entry(pos->member.next, typeof(*pos), member);    \
&pos->member != (head);                     \
pos = n, n = list_entry(n->member.next, typeof(*n), member))

#endif