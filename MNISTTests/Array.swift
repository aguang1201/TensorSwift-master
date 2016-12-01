extension Array {
    func grouped(_ count: Int) -> [[Element]] {
        var result: [[Element]] = []
        var group: [Element] = []
        for element in self {
            group.append(element)
            if group.count == count {
                result.append(group)
                group = []
            }
        }
        return result
    }
}
