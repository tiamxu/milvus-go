package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"milvus_demo/milvus"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

var (
	milvusAddr = "10.18.150.1:19530"
)

const (
	nEntities, dim                 = 1000, 128
	collectionName                 = "gosdk_index_example"
	msgFmt                         = "==== %s ====\n"
	idCol, randomCol, embeddingCol = "ID", "random", "embeddings"
	topK                           = 3
)

func main() {
	ctx := context.Background()
	client, err := client.NewClient(ctx, client.Config{
		Address: milvusAddr,
	})
	defer client.Close()
	films, err := loadFilmCSV()
	if err != nil {
		log.Fatal("failed to load film data csv:", err.Error())
	}
	searchFilm := films[0] // use first fim to search
	vector := entity.FloatVector(searchFilm.Vector[:])
	idTitle := make(map[int64]string)

	sp, _ := entity.NewIndexFlatSearchParam()
	start := time.Now()
	sr, err := client.Search(ctx, collectionName, []string{}, "Year > 1990", []string{"ID"}, []entity.Vector{vector}, "Vector",
		entity.L2, 10, sp)
	if err != nil {
		log.Fatal("fail to search collection:", err.Error())
	}
	log.Println("search without index time elapsed:", time.Since(start))
	for _, result := range sr {
		var idColumn *entity.ColumnInt64
		for _, field := range result.Fields {
			if field.Name() == "ID" {
				c, ok := field.(*entity.ColumnInt64)
				if ok {
					idColumn = c
				}
			}
		}
		if idColumn == nil {
			log.Fatal("result field not math")
		}
		for i := 0; i < result.ResultCount; i++ {
			id, err := idColumn.ValueByIdx(i)
			if err != nil {
				log.Fatal(err.Error())
			}
			title := idTitle[id]
			fmt.Printf("file id: %d title: %s scores: %f\n", id, title, result.Scores[i])
		}
	}
	// schema := &entity.Schema{
	// 	CollectionName: collectionName,
	// 	Description:    "测试 搜索",
	// 	Fields: []*entity.Field{
	// 		{
	// 			Name:       "book_id",
	// 			DataType:   entity.FieldTypeInt64,
	// 			PrimaryKey: true,
	// 			AutoID:     false,
	// 		},
	// 		{
	// 			Name:       "word_count",
	// 			DataType:   entity.FieldTypeInt64,
	// 			PrimaryKey: false,
	// 			AutoID:     false,
	// 		},
	// 		{
	// 			Name:     "book_intro",
	// 			DataType: entity.FieldTypeFloatVector,
	// 			TypeParams: map[string]string{
	// 				"dim": "2",
	// 			},
	// 		},
	// 	},
	// 	EnableDynamicField: true,
	// }

}

func insert() {
	ctx := context.Background()
	client, err := milvus.NewMilvusClient(milvusAddr)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()
	schema := entity.NewSchema().WithName(collectionName).WithDescription("this is the example collection for indexing").
		WithField(entity.NewField().WithName("ID").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("Year").WithDataType(entity.FieldTypeInt32)).
		WithField(entity.NewField().WithName("Vector").WithDataType(entity.FieldTypeFloatVector).WithDim(8))

	client.CreateCollection(ctx, collectionName, schema)
	films, err := loadFilmCSV()
	if err != nil {
		log.Fatal("failed to load film data csv:", err.Error())
	}
	ids := make([]int64, 0, len(films))
	years := make([]int32, 0, len(films))
	vectors := make([][]float32, 0, len(films))
	// string field is not supported yet
	idTitle := make(map[int64]string)
	for idx, film := range films {
		ids = append(ids, film.ID)
		idTitle[film.ID] = film.Title
		years = append(years, film.Year)
		vectors = append(vectors, films[idx].Vector[:]) // prevent same vector
	}
	idColumn := entity.NewColumnInt64("ID", ids)
	yearColumn := entity.NewColumnInt32("Year", years)
	vectorColumn := entity.NewColumnFloatVector("Vector", 8, vectors)

	if err = client.Insert(ctx, collectionName, "", idColumn, yearColumn, vectorColumn); err != nil {
		log.Fatal("插入数据失败:", err.Error())
	}
	log.Println("insert completed")

	idx, err := entity.NewIndexIvfFlat(entity.L2, 128)
	if err != nil {
		fmt.Printf("failed to create ivf flat index: %s", err)
	}
	if err = client.CreateIndex(ctx, collectionName, "Vector", idx); err != nil {
		log.Fatal("创建索引失败:", err.Error())
	}

	sidx := entity.NewScalarIndex()
	if err := client.CreateIndex(ctx, collectionName, "Year", sidx); err != nil {
		log.Fatal("failed to create scalar index", err.Error())
	}
	if err = client.LoadCollection(ctx, collectionName); err != nil {
		log.Fatal("加载集合失败:", err.Error())
	}
}

type film struct {
	ID     int64
	Title  string
	Year   int32
	Vector [8]float32 // fix length array
}

func loadFilmCSV() ([]film, error) {
	f, err := os.Open("./films.csv") // assume you are in examples/insert folder, if not, please change the path
	if err != nil {
		return []film{}, err
	}
	r := csv.NewReader(f)
	raw, err := r.ReadAll()
	if err != nil {
		return []film{}, err
	}
	films := make([]film, 0, len(raw))
	for _, line := range raw {
		if len(line) < 4 { // insuffcient column
			continue
		}
		fi := film{}
		// ID
		v, err := strconv.ParseInt(line[0], 10, 64)
		if err != nil {
			continue
		}
		fi.ID = v
		// Title
		fi.Title = line[1]
		// Year
		v, err = strconv.ParseInt(line[2], 10, 64)
		if err != nil {
			continue
		}
		fi.Year = int32(v)
		// Vector
		vectorStr := strings.ReplaceAll(line[3], "[", "")
		vectorStr = strings.ReplaceAll(vectorStr, "]", "")
		parts := strings.Split(vectorStr, ",")
		if len(parts) != 8 { // dim must be 8
			continue
		}
		for idx, part := range parts {
			part = strings.TrimSpace(part)
			v, err := strconv.ParseFloat(part, 32)
			if err != nil {
				continue
			}
			fi.Vector[idx] = float32(v)
		}
		films = append(films, fi)
	}
	return films, nil
}
