# SIG2025
Signal and image processing for KU 2025 

## Project Structure

* a${week_number}.tex 是每周的report; **页数限制在12页内，注意调整图片尺寸，文字写多了找gpt精简一下**

* pics 存放tex中用到的图片

* code_week${week_number} 是每周的代码

文件命名参考已有文件便于管理和找到，建议重命名方便维护 比如图片a2-1.1表示a2.tex中section 1.1引用的图片

## Usage

```shell
git clone ${url}
# change code and .tex
git add ${path to your changes}
git commit -m "your modification message"
git pull
# solve conflicts
git push

```
